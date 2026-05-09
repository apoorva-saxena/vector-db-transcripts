import 'dotenv/config';
import { InferenceClient } from '@huggingface/inference';
import { Pool } from 'pg';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

const hf = new InferenceClient(process.env.HF_TOKEN);

const pool = new Pool({
  host: 'localhost',
  port: 5433,
  database: 'vectordb',
  user: 'postgres',
  password: 'postgres',
});

const EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const CHUNK_SIZE = 4;

interface DialogueTurn {
  speaker: string;
  text: string;
}

interface Chunk {
  index: number;
  text: string;
  speakers: string[];
}

function parseMeta(lines: string[]): {
  title: string;
  category: string;
  speakers: string[];
  recorded_date: string;
  bodyStart: number;
} {
  let title = '';
  let category = '';
  let speakers: string[] = [];
  let recorded_date = '';
  let i = 0;

  for (; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) break;
    if (line.startsWith('Topic:'))
      title = line.slice('Topic:'.length).trim();
    else if (line.startsWith('Date:'))
      recorded_date = line.slice('Date:'.length).trim();
    else if (line.startsWith('Category:'))
      category = line.slice('Category:'.length).trim();
    else if (line.startsWith('Speakers:'))
      // "Priya (frontend lead), Marcus (senior engineer)" -> ["Priya", "Marcus"]
      speakers = line
        .slice('Speakers:'.length)
        .split(',')
        .map((s) => s.trim().split(' ')[0]);
  }

  return { title, category, speakers, recorded_date, bodyStart: i + 1 };
}

function parseDialogue(lines: string[], speakers: string[]): DialogueTurn[] {
  const turns: DialogueTurn[] = [];
  let currentSpeaker = '';
  let currentText = '';

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const speaker = speakers.find((s) => trimmed.startsWith(`${s}:`));
    if (speaker) {
      if (currentSpeaker) turns.push({ speaker: currentSpeaker, text: currentText.trim() });
      currentSpeaker = speaker;
      currentText = trimmed.slice(speaker.length + 1).trim();
    } else if (currentSpeaker) {
      currentText += ' ' + trimmed;
    }
  }

  if (currentSpeaker) turns.push({ speaker: currentSpeaker, text: currentText.trim() });
  return turns;
}

function buildChunks(turns: DialogueTurn[]): Chunk[] {
  const chunks: Chunk[] = [];
  for (let i = 0; i < turns.length; i += CHUNK_SIZE) {
    const group = turns.slice(i, i + CHUNK_SIZE);
    chunks.push({
      index: chunks.length,
      text: group.map((t) => `${t.speaker}: ${t.text}`).join('\n\n'),
      speakers: [...new Set(group.map((t) => t.speaker))],
    });
  }
  return chunks;
}

async function embed(text: string): Promise<number[]> {
  const raw = await hf.featureExtraction({ model: EMBED_MODEL, inputs: text });
  if (typeof (raw as unknown as number[])[0] === 'number') return raw as unknown as number[];
  return (raw as unknown as number[][])[0];
}

async function ingest(filepath: string) {
  const filename = filepath.split('/').pop()!;
  const lines = readFileSync(filepath, 'utf-8').split('\n');

  const { title, category, speakers, recorded_date, bodyStart } = parseMeta(lines);
  const turns = parseDialogue(lines.slice(bodyStart), speakers);
  const chunks = buildChunks(turns);

  console.log(`\n📄 ${filename}  (${recorded_date} · ${category})`);
  console.log(`   ${speakers.join(', ')} · ${turns.length} turns → ${chunks.length} chunks`);

  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    const { rows } = await client.query(
      `INSERT INTO transcripts (filename, title, category, speakers, recorded_date)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (filename) DO UPDATE SET
         title         = EXCLUDED.title,
         category      = EXCLUDED.category,
         speakers      = EXCLUDED.speakers,
         recorded_date = EXCLUDED.recorded_date
       RETURNING id`,
      [filename, title, category, speakers, recorded_date]
    );
    const transcriptId = rows[0].id;

    await client.query('DELETE FROM transcript_chunks WHERE transcript_id = $1', [transcriptId]);

    for (const chunk of chunks) {
      process.stdout.write(`   chunk ${chunk.index + 1}/${chunks.length} ...`);
      const embedding = await embed(chunk.text);

      await client.query(
        `INSERT INTO transcript_chunks
           (transcript_id, chunk_index, chunk_text, speakers_in_chunk, embed_model, embedding, embedded_at)
         VALUES ($1, $2, $3, $4, $5, $6::vector, now())`,
        [
          transcriptId,
          chunk.index,
          chunk.text,
          chunk.speakers,
          EMBED_MODEL,
          `[${embedding.join(',')}]`,
        ]
      );
      process.stdout.write(' ✓\n');
    }

    await client.query('COMMIT');
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}

async function main() {
  console.log('🚀 Ingesting transcripts...');
  const dir = join(process.cwd(), 'transcripts');
  const files = readdirSync(dir)
    .filter((f) => f.endsWith('.txt'))
    .map((f) => join(dir, f));

  for (const file of files) {
    await ingest(file);
  }

  console.log('\n✅ Done.');
  await pool.end();
}

main().catch(console.error);
