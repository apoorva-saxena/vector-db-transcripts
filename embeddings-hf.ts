import 'dotenv/config';
import { InferenceClient } from '@huggingface/inference';

const hf = new InferenceClient(process.env.HF_TOKEN);

type FeatureTensor = Awaited<ReturnType<InferenceClient['featureExtraction']>>;

/** Normalize HF feature extraction output to a single embedding vector. */
function toEmbeddingVector(raw: FeatureTensor): number[] {
  if (!raw.length) return [];
  const first = raw[0];
  if (typeof first === 'number') {
    return raw as number[];
  }
  if (Array.isArray(first)) {
    const inner = first[0];
    if (typeof inner === 'number') {
      return first as number[];
    }
    const rows = first as number[][];
    if (!rows.length) return [];
    const dim = rows[0].length;
    const sum = new Array(dim).fill(0);
    for (const row of rows) {
      for (let i = 0; i < dim; i++) sum[i] += row[i];
    }
    return sum.map((v) => v / rows.length);
  }
  return [];
}

async function generateEmbeddings() {
  console.log('🤗 Using Hugging Face Free Embeddings\n');

  const texts = [
    'The cat sat on the mat',
    'A feline rested on the rug',
    'JavaScript is a programming language'
  ];

  for (const text of texts) {
    const raw = await hf.featureExtraction({
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      inputs: text
    });
    const embedding = toEmbeddingVector(raw);

    console.log(`Text: "${text}"`);
    console.log(`Embedding dimensions: ${embedding.length}`);
    console.log(`First 10: [${embedding.slice(0, 10).map((n) => n.toFixed(3)).join(', ')}]`);
    console.log('');
  }
}

generateEmbeddings();
