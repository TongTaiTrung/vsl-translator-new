
import { cleanCompletion, composeMessages, DEFAULT_MODEL, naiveSentence } from "@/lib/sentence";

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const MAX_WORDS = 64;
const MAX_LABEL_LENGTH = 64;
const MODEL_RE = /^[\w.:/-]{1,100}$/;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function json(body: unknown, status = 200) {
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: Request) {
  const key = request.headers.get("x-groq-key")?.trim();
  if (!key) return json({ error: "Chưa nhập Groq API key" }, 400);

  const body = await request.json().catch(() => null);
  const words: unknown = body?.words;
  if (
    !Array.isArray(words) || words.length === 0 || words.length > MAX_WORDS ||
    !words.every((w) => typeof w === "string" && w.length > 0 && w.length <= MAX_LABEL_LENGTH)
  ) {
    return json({ error: `'words' phải là mảng 1–${MAX_WORDS} nhãn` }, 400);
  }
  const model = typeof body?.model === "string" && body.model.trim() ? body.model.trim() : DEFAULT_MODEL;
  if (!MODEL_RE.test(model)) return json({ error: "Tên model không hợp lệ" }, 400);

  const payload: Record<string, unknown> = {
    model,
    messages: composeMessages(words),
    temperature: 0.2,
    max_completion_tokens: 200,
  };
  if (model.includes("qwen3")) payload.reasoning_effort = "none";

  let response: Response;
  try {
    response = await fetch(GROQ_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(15_000),
      cache: "no-store",
    });
  } catch (err) {
    return json({ error: `Không kết nối được Groq: ${(err as Error).message}` }, 502);
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    const message =
      response.status === 401
        ? "Groq API key không hợp lệ"
        : detail?.error?.message || `Groq API lỗi ${response.status}`;
    return json({ error: message }, response.status === 401 ? 401 : 502);
  }

  const data = await response.json();
  const sentence = cleanCompletion(data?.choices?.[0]?.message?.content ?? "");
  return sentence
    ? json({ sentence, status: "ok", model })
    : json({ sentence: naiveSentence(words), status: "fallback", model });
}
