import { getLlama, LlamaLogLevel, resolveModelFile } from "node-llama-cpp";

async function readRequest() {
  return await new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      try {
        resolve(JSON.parse(data));
      } catch (error) {
        reject(error);
      }
    });
    process.stdin.on("error", reject);
  });
}

async function embedTexts({ model, model_cache_dir: modelCacheDir, texts }) {
  if (!Array.isArray(texts)) {
    throw new Error("Expected `texts` to be an array.");
  }
  if (typeof model !== "string" || model.length === 0) {
    throw new Error("Expected `model` to be a non-empty string.");
  }
  if (typeof modelCacheDir !== "string" || modelCacheDir.length === 0) {
    throw new Error("Expected `model_cache_dir` to be a non-empty string.");
  }

  const llama = await getLlama({
    build: "autoAttempt",
    logLevel: LlamaLogLevel.error,
  });
  try {
    const modelPath = await resolveModelFile(model, modelCacheDir);
    const loadedModel = await llama.loadModel({ modelPath });
    const context = await loadedModel.createEmbeddingContext();
    const vectors = [];
    for (const text of texts) {
      const embedding = await context.getEmbeddingFor(text);
      vectors.push(Array.from(embedding.vector));
    }
    process.stdout.write(JSON.stringify({ vectors }));
  } finally {
    await llama.dispose();
  }
}

try {
  const request = await readRequest();
  await embedTexts(request);
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(message + "\n");
  process.exitCode = 1;
}
