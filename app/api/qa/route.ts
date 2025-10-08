import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

// URL k model.onnx z GitHub Release
const MODEL_URL = 'https://github.com/mnaukohutka/ai-backend-vercel/releases/download/v1.0/model.onnx';
const MODEL_CACHE = '/tmp/model.onnx';

let session: ort.InferenceSession | null = null;
let vocabData: any = null;
let modelLoading: Promise<void> | null = null;

// Stažení modelu z GitHub Release
async function downloadModel(): Promise<void> {
  if (fs.existsSync(MODEL_CACHE)) {
    console.log('✅ Model už je v cache');
    return;
  }

  return new Promise((resolve, reject) => {
    console.log('📥 Stahuji model z GitHub Release...');
    const file = fs.createWriteStream(MODEL_CACHE);
    
    https.get(MODEL_URL, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // GitHub redirect
        https.get(response.headers.location!, (redirectResponse) => {
          redirectResponse.pipe(file);
          file.on('finish', () => {
            file.close();
            console.log('✅ Model stažen (124 MB)');
            resolve();
          });
        }).on('error', reject);
      } else {
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          console.log('✅ Model stažen (124 MB)');
          resolve();
        });
      }
    }).on('error', (err) => {
      fs.unlink(MODEL_CACHE, () => {});
      reject(err);
    });
  });
}

// Načtení ONNX session
async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session;

  // Pokud už nějaký thread stahuje, počkej
  if (modelLoading) {
    await modelLoading;
    if (session) return session;
  }

  // Začni stahovat
  modelLoading = downloadModel();
  await modelLoading;
  modelLoading = null;
  
  console.log('🤖 Načítám ONNX session...');
  session = await ort.InferenceSession.create(MODEL_CACHE, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });
  console.log('✅ Session připravena');
  
  return session;
}

// Načtení vocab.json
function getVocab() {
  if (!vocabData) {
    const vocabPath = path.join(process.cwd(), 'public/model-releases/vocab.json');
    vocabData = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));
  }
  return vocabData;
}

// Tokenizace
function tokenize(text: string): number[] {
  const vocab = getVocab();
  const words = text
    .toLowerCase()
    .replace(/[^\w\sáčďéěíňóřšťúůýž]/g, '')
    .split(/\s+/)
    .filter(w => w);
  
  const tokens: number[] = [];
  for (const word of words) {
    tokens.push(vocab[word] ?? 0); // 0 = UNK
  }
  return tokens;
}

// Detokenizace
function detokenize(tokenIds: number[]): string {
  const vocab = getVocab();
  const reverseVocab: { [key: number]: string } = {};
  
  for (const [word, id] of Object.entries(vocab)) {
    reverseVocab[id as number] = word;
  }
  
  return tokenIds
    .map(id => reverseVocab[id])
    .filter(w => w && !['<unk>', '<pad>', '<s>', '</s>'].includes(w))
    .join(' ')
    .trim();
}

// POST endpoint
export async function POST(req: Request) {
  const startTime = Date.now();
  
  try {
    const { question } = await req.json();
    
    if (!question || typeof question !== 'string') {
      return Response.json({ 
        error: 'Parametr "question" je povinný' 
      }, { status: 400 });
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`📝 Otázka: ${question}`);
    
    // Načti model
    const model = await getSession();
    
    // Tokenizuj
    const prompt = `Otázka: ${question}\nOdpověď:`;
    const inputTokens = tokenize(prompt);
    console.log(`🔤 Input tokeny: ${inputTokens.length}`);
    
    // Padding na 512
    const maxLength = 512;
    const paddedTokens = [...inputTokens];
    
    if (paddedTokens.length > maxLength) {
      paddedTokens.length = maxLength;
    } else {
      while (paddedTokens.length < maxLength) {
        paddedTokens.push(0);
      }
    }
    
    // Vytvoř tensory
    const inputIds = new ort.Tensor(
      'int64',
      BigInt64Array.from(paddedTokens.map(t => BigInt(t))),
      [1, maxLength]
    );
    
    const attentionMask = new ort.Tensor(
      'int64',
      BigInt64Array.from(paddedTokens.map(t => t === 0 ? BigInt(0) : BigInt(1))),
      [1, maxLength]
    );
    
    // Inference
    console.log('🤖 Inference...');
    const inferenceStart = Date.now();
    
    const results = await model.run({
      input_ids: inputIds,
      attention_mask: attentionMask
    });
    
    const inferenceTime = Date.now() - inferenceStart;
    console.log(`✅ Inference hotová (${inferenceTime}ms)`);
    
    // Zpracuj výstup - greedy dekódování
    const logits = results.logits;
    const logitsData = logits.data as Float32Array;
    const vocabSize = Object.keys(getVocab()).length;
    
    const startIdx = inputTokens.length;
    const maxNewTokens = 100;
    const outputTokens: number[] = [];
    
    for (let i = 0; i < maxNewTokens; i++) {
      const tokenPos = startIdx + i;
      if (tokenPos >= maxLength) break;
      
      const offset = tokenPos * vocabSize;
      let maxProb = -Infinity;
      let maxToken = 0;
      
      for (let t = 0; t < Math.min(vocabSize, 10000); t++) {
        const prob = logitsData[offset + t];
        if (prob > maxProb) {
          maxProb = prob;
          maxToken = t;
        }
      }
      
      if (maxToken === 0 || maxToken === 2) break;
      outputTokens.push(maxToken);
    }
    
    const answer = detokenize(outputTokens);
    const totalTime = Date.now() - startTime;
    
    console.log(`💬 Odpověď: ${answer}`);
    console.log(`⏱️  Čas: ${totalTime}ms`);
    console.log(`${'='.repeat(60)}\n`);
    
    return Response.json({
      question: question,
      answer: answer || 'Model negeneroval odpověď',
      metadata: {
        model: 'Kvantizovaný ONNX (124 MB)',
        input_tokens: inputTokens.length,
        output_tokens: outputTokens.length,
        inference_time_ms: inferenceTime,
        total_time_ms: totalTime
      }
    });
    
  } catch (error: any) {
    console.error('❌ Chyba:', error);
    return Response.json({
      error: error.message,
      type: error.constructor.name
    }, { status: 500 });
  }
}

// GET endpoint pro status
export async function GET() {
  const modelCached = fs.existsSync(MODEL_CACHE);
  const modelSize = modelCached ? 
    (fs.statSync(MODEL_CACHE).size / (1024 * 1024)).toFixed(2) + ' MB' : 
    'není stažen';
  
  return Response.json({
    status: 'online',
    model: 'Kvantizovaný czech-gpt2-finetuned-qa',
    model_size: '124 MB',
    model_cached: modelCached,
    cache_size: modelSize,
    endpoint: '/api/qa',
    usage: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { question: 'tvoje otázka' }
    },
    note: modelCached ? 
      'Model je v cache, requests budou rychlé' : 
      'První request stáhne model (~30s)'
  });
}
