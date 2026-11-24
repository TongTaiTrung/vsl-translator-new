// Bản TypeScript của src/app/sentence.py — giữ prompt, chú giải và cách ghép giống hệt bản Python.

export const DEFAULT_MODEL = "qwen/qwen3.8-27b";

// Nghĩa của các nhãn thô không tự đọc được. Nhãn khác: "_" được đổi thành khoảng trắng.
export const DEFAULT_GLOSSARY: Record<string, string> = {
  toi: "tôi",
  ten_la: "tên là",
  age: "tuổi",
};

const SYSTEM_PROMPT_TEMPLATE = `Bạn là bộ hậu xử lý của hệ thống dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang tiếng Việt viết.

ĐẦU VÀO là chuỗi NHÃN THÔ do mô hình nhận dạng ký hiệu đưa ra, đúng thứ tự người dùng ký, ngăn cách bởi " | ".
Đặc điểm của nhãn thô:
- Có thể viết không dấu hoặc dùng "_" thay khoảng trắng. Bảng chú giải bên dưới cho biết nghĩa của các nhãn đã biết.
- Mục dạng "[đánh vần: ...]" là các chữ cái đánh vần bằng ngón tay liên tiếp, dùng cho tên riêng hoặc từ chưa có ký hiệu riêng (ví dụ T-R-U-N-G → "Trung").
- Trong mục đánh vần có thể có DẤU THANH dạng "(dấu sắc)", "(dấu huyền)", "(dấu hỏi)", "(dấu ngã)", "(dấu nặng)". Dấu thanh được ký NGAY SAU các chữ cái của âm tiết nó thuộc về: đặt dấu đó lên âm tiết đứng liền trước, đúng vị trí theo chính tả tiếng Việt (B-I-(dấu nặng) → "bị", T-O-A-N-(dấu sắc) → "toán"). Dấu thanh cũng là dấu hiệu kết thúc một âm tiết.
- Trong mục đánh vần còn có DẤU CHỮ dạng "(dấu â,ô,ê)" (dấu mũ) và "(dấu ơ,ư)" (dấu móc). Dấu chữ biến nguyên âm đứng gần nhất phía trước nó thành chữ có dấu: "(dấu â,ô,ê)" đổi a→â, o→ô, e→ê; "(dấu ơ,ư)" đổi o→ơ, u→ư (T-H-O-(dấu ơ,ư) → "Thơ", C-O-(dấu â,ô,ê)-N-G → "công", H-O-(dấu ơ,ư)-N → "hơn"). Dấu chữ là một phần của chữ cái, KHÔNG phải một từ và KHÔNG kết thúc âm tiết.
- MỘT mục đánh vần có thể chứa NHIỀU âm tiết/từ viết liền, không có ký hiệu ngăn cách. Hãy tách chuỗi chữ cái thành các âm tiết tiếng Việt hợp lệ sao cho ghép lại có nghĩa nhất (ví dụ M-I-N-H-B-I-(dấu nặng)-N-G-U → "Minh bị ngu"). Viết hoa chữ đầu của tên riêng; các từ thường viết thường.
- BẮT BUỘC giữ mọi dấu thanh đã được ký, kể cả khi bỏ dấu cũng ra một tên/từ có thật (H-O-A-(dấu huyền) là "Hòa", không phải "Hoa"). Âm tiết không có dấu thanh đi kèm thì giữ thanh ngang, trừ khi chắc chắn đó là tên/từ tiếng Việt phổ biến bị thiếu dấu.
- TUYỆT ĐỐI không viết nguyên văn tên dấu ("dấu ơ,ư", "dấu â,ô,ê", "dấu nặng"...) vào kết quả, kể cả khi nhãn dấu đứng riêng ngoài mục đánh vần: luôn đặt dấu đó lên chữ cái liền trước.
- Ngữ pháp ký hiệu khác tiếng Việt nói: thường lược hư từ, trợ từ và từ "là", có thể đưa chủ đề lên trước, từ để hỏi thường đứng cuối.
- Mô hình nhận dạng đôi khi lặp nhãn; bỏ những lần lặp không có nghĩa.

NHIỆM VỤ: ghép chuỗi nhãn thành câu tiếng Việt tự nhiên, đúng chính tả, đủ dấu thanh và dấu câu.
QUY TẮC:
1. Giữ đúng ý và trình tự ý của người ký. Không bịa thêm thông tin, tên, số hay chi tiết không có trong nhãn.
2. Được thêm hư từ, trợ từ, từ nối cần thiết (là, của, ở, thì, ạ, và...) và sắp xếp lại cho đúng ngữ pháp.
3. Nếu chuỗi chưa thành câu trọn vẹn (người dùng đang ký dở), trả về cụm từ tự nhiên nhất, không tự thêm ý.
4. Chỉ trả về đúng kết quả trên một dòng: không giải thích, không ngoặc kép, không tiền tố kiểu "Câu:".
5. Kết quả LUÔN kết thúc bằng dấu câu: dấu chấm "." (hoặc "?" nếu là câu hỏi, "!" nếu là câu cảm thán), kể cả khi chỉ là một cụm từ.

BẢNG CHÚ GIẢI NHÃN:
{glossary}

VÍ DỤ:
Nhãn: xin chào | toi | ten_la | [đánh vần: T-R-U-N-G]
Kết quả: Xin chào, tôi tên là Trung.
Nhãn: toi | địa chỉ
Kết quả: Địa chỉ của tôi.
Nhãn: cảm ơn | cảm ơn
Kết quả: Cảm ơn.
Nhãn: [đánh vần: M-I-N-H-B-I-(dấu nặng)-N-G-U]
Kết quả: Minh bị ngu.
Nhãn: xin chào | toi | ten_la | [đánh vần: T-H-O-(dấu ơ,ư)]
Kết quả: Xin chào, tôi tên là Thơ.
`;

type Group = string | string[];

// Nhãn dấu thanh ký trong lúc đánh vần → dấu kết hợp Unicode tương ứng.
export const TONE_MARKS: Record<string, string> = {
  "dấu sắc": "\u0301",
  "dấu huyền": "\u0300",
  "dấu hỏi": "\u0309",
  "dấu ngã": "\u0303",
  "dấu nặng": "\u0323",
};
const VOWELS = new Set([..."aăâeêioôơuưy"]);

// Nhãn dấu chữ (mũ, móc) → nguyên âm gốc được đổi thành chữ có dấu.
export const VOWEL_MARKS: Record<string, Record<string, string>> = {
  "dấu â,ô,ê": { a: "â", o: "ô", e: "ê" },
  "dấu ơ,ư": { o: "ơ", u: "ư" },
};

const isLetter = (label: string) => [...label].length === 1 && /^\p{L}$/u.test(label);
// Tên lớp trong dữ liệu khác tên chuẩn của dấu (lớp "dấu sắt" được thu với tên gõ sai).
export const MARK_ALIASES: Record<string, string> = {
  "dấu sắt": "dấu sắc",
};
const toneKey = (label: string) => {
  const key = label.normalize("NFC").trim().toLowerCase();
  return MARK_ALIASES[key] ?? key;
};
const isTone = (label: string) => toneKey(label) in TONE_MARKS;
const isVowelMark = (label: string) => toneKey(label) in VOWEL_MARKS;
const isMark = (label: string) => isTone(label) || isVowelMark(label);
const spellToken = (label: string) => (isMark(label) ? `(${toneKey(label)})` : label.toUpperCase());

/** Đổi nguyên âm gốc gần cuối nhất (a/o/e/u) thành chữ có dấu mũ/móc, giữ nguyên dấu thanh nếu đã có. */
function applyVowelMark(text: string, label: string): string {
  const mapping = VOWEL_MARKS[toneKey(label)];
  const chars = text.normalize("NFD");
  for (let i = chars.length - 1; i >= 0; i--) {
    if (chars[i] in mapping && !"\u0302\u031b\u0306".includes(chars[i + 1] ?? "x")) {
      return (chars.slice(0, i) + mapping[chars[i]].normalize("NFD") + chars.slice(i + 1)).normalize("NFC");
    }
  }
  return text;
}

// "ưo" (dạng NFD) → "ươ"
const fixUo = (run: string) => run.normalize("NFD").replace(/u\u031bo(?!\u031b)/g, "u\u031bo\u031b").normalize("NFC");

// Âm tiết tiếng Việt (chưa dấu thanh): phụ âm đầu + 1–3 nguyên âm + phụ âm cuối. Dùng để tách chữ đánh vần liền nhau.
const SYLLABLE_RE =
  /^(ngh|ng|gh|gi|kh|nh|ph|th|tr|ch|qu|[bcdđghklmnprstvx])?([aăâeêioôơuưy]{1,3})(ng|nh|ch|[cmnpt])?$/u;
const MAX_SYLLABLE = 7; // "nghiêng"

/** Tách chuỗi chữ cái liền thành ít âm tiết hợp lệ nhất (ưu tiên âm tiết trước dài hơn); không tách được thì giữ nguyên. */
function splitSyllables(run: string): string[] {
  const chars = [...run];
  const best: (string[] | null)[] = [[], ...chars.map(() => null)];
  for (let end = 1; end <= chars.length; end++) {
    for (let start = end - 1; start >= Math.max(0, end - MAX_SYLLABLE); start--) {
      const prev = best[start];
      const piece = chars.slice(start, end).join("");
      if (prev && SYLLABLE_RE.test(piece)) {
        const current = best[end];
        if (!current || prev.length + 1 < current.length) best[end] = [...prev, piece];
      }
    }
  }
  return best[chars.length] ?? [run];
}

const insertAt = (chars: string[], pos: number, mark: string) =>
  [...chars.slice(0, pos + 1), mark, ...chars.slice(pos + 1)].join("").normalize("NFC");

/** Đặt dấu thanh lên đúng nguyên âm theo chính tả (kiểu cũ: hòa, thúy). */
function applyTone(syllable: string, mark: string): string {
  const match = SYLLABLE_RE.exec(syllable);
  if (!match) {
    const chars = [...syllable];
    const idx = chars.findLastIndex((c) => VOWELS.has(c));
    return idx >= 0 ? insertAt(chars, idx, mark) : syllable;
  }
  const [, onset = "", vowelText, coda = ""] = match;
  const vowels = [...vowelText];
  const special = vowels.findLastIndex((c) => "ăâêôơư".includes(c));
  const pos = special >= 0 ? special : coda ? vowels.length - 1 : vowels.length === 3 ? 1 : 0;
  return onset + insertAt(vowels, pos, mark) + coda;
}

/** Câu dự phòng cho một nhóm đánh vần: tách âm tiết, đổi chữ theo dấu mũ/móc, đặt dấu thanh lên âm tiết ngay trước dấu. */
function spelledText(items: readonly string[]): string {
  const words: string[] = [];
  let run = "";
  for (const item of items) {
    if (isVowelMark(item)) {
      if (run) run = applyVowelMark(run, item);
      else if (words.length) words[words.length - 1] = applyVowelMark(words[words.length - 1], item); // dấu chữ ký sau dấu thanh
      continue;
    }
    if (!isTone(item)) {
      run += item.toLowerCase();
      continue;
    }
    if (run) {
      const syllables = splitSyllables(fixUo(run));
      syllables[syllables.length - 1] = applyTone(syllables[syllables.length - 1], TONE_MARKS[toneKey(item)]);
      words.push(...syllables);
      run = "";
    }
  }
  if (run) words.push(...splitSyllables(fixUo(run)));
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function labelText(label: string, glossary = DEFAULT_GLOSSARY): string {
  return glossary[label] ?? label.replaceAll("_", " ");
}

export function buildSystemPrompt(glossary = DEFAULT_GLOSSARY): string {
  const lines = Object.entries(glossary)
    .map(([raw, text]) => `- "${raw}" = "${text}"`)
    .join("\n");
  return SYSTEM_PROMPT_TEMPLATE.replace("{glossary}", lines || "- (không có)");
}

/** Gom các chữ cái đơn liên tiếp (đánh vần) thành một nhóm; dấu thanh và dấu chữ ngay sau chữ cái thuộc nhóm đó. */
export function groupLabels(words: readonly string[]): Group[] {
  const groups: Group[] = [];
  for (const word of words) {
    const last = groups[groups.length - 1];
    if (isLetter(word) || (isMark(word) && Array.isArray(last))) {
      if (Array.isArray(last)) last.push(word);
      else groups.push([word]);
    } else {
      groups.push(word);
    }
  }
  return groups;
}

export function formatLabels(words: readonly string[]): string {
  return groupLabels(words)
    .map((item) =>
      Array.isArray(item) ? `[đánh vần: ${item.map(spellToken).join("-")}]` : item,
    )
    .join(" | ");
}

/** Câu dự phòng khi không gọi được LLM: dịch nhãn theo chú giải và ghép chữ cái. */
export function naiveSentence(words: readonly string[], glossary = DEFAULT_GLOSSARY): string {
  const text = groupLabels(words)
    .map((item) => (Array.isArray(item) ? spelledText(item) : labelText(item, glossary)))
    .join(" ")
    .trim();
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
}

export function cleanCompletion(text: string): string {
  let out = (text || "").replace(/<think>[\s\S]*?(<\/think>|$)/gi, "").trim();
  out = out.replace(/^(kết quả|câu)\s*:\s*/i, "");
  const line = out
    .split(/\r?\n/)
    .map((part) => part.trim())
    .find(Boolean);
  const sentence = (line ?? "").replace(/^["“”']+|["“”']+$/g, "");
  // Luôn kết thúc câu bằng dấu câu, kể cả khi LLM quên.
  return sentence && !/[.?!…]$/.test(sentence) ? `${sentence}.` : sentence;
}

export function composeMessages(words: readonly string[]) {
  return [
    { role: "system", content: buildSystemPrompt() },
    { role: "user", content: `Nhãn: ${formatLabels(words)}\nKết quả:` },
  ];
}
