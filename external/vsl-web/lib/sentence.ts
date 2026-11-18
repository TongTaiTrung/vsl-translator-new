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
- Mục dạng "[đánh vần: T-R-U-N-G]" là các chữ cái đánh vần bằng ngón tay liên tiếp. Ghép chúng thành MỘT từ, thường là tên riêng hoặc từ chưa có ký hiệu riêng (ví dụ T-R-U-N-G → "Trung"). Viết hoa chữ đầu nếu là tên riêng. Chỉ thêm dấu thanh khi chắc chắn đó là tên/từ tiếng Việt phổ biến; nếu không, giữ nguyên các chữ cái.
- Ngữ pháp ký hiệu khác tiếng Việt nói: thường lược hư từ, trợ từ và từ "là", có thể đưa chủ đề lên trước, từ để hỏi thường đứng cuối.
- Mô hình nhận dạng đôi khi lặp nhãn; bỏ những lần lặp không có nghĩa.

NHIỆM VỤ: ghép chuỗi nhãn thành câu tiếng Việt tự nhiên, đúng chính tả, đủ dấu thanh và dấu câu.
QUY TẮC:
1. Giữ đúng ý và trình tự ý của người ký. Không bịa thêm thông tin, tên, số hay chi tiết không có trong nhãn.
2. Được thêm hư từ, trợ từ, từ nối cần thiết (là, của, ở, thì, ạ, và...) và sắp xếp lại cho đúng ngữ pháp.
3. Nếu chuỗi chưa thành câu trọn vẹn (người dùng đang ký dở), trả về cụm từ tự nhiên nhất, không tự thêm ý.
4. Chỉ trả về đúng kết quả trên một dòng: không giải thích, không ngoặc kép, không tiền tố kiểu "Câu:".

BẢNG CHÚ GIẢI NHÃN:
{glossary}

VÍ DỤ:
Nhãn: xin chào | toi | ten_la | [đánh vần: T-R-U-N-G]
Kết quả: Xin chào, tôi tên là Trung.
Nhãn: toi | địa chỉ
Kết quả: Địa chỉ của tôi
Nhãn: cảm ơn | cảm ơn
Kết quả: Cảm ơn.
`;

type Group = string | string[];

const isLetter = (label: string) => [...label].length === 1 && /^\p{L}$/u.test(label);

export function labelText(label: string, glossary = DEFAULT_GLOSSARY): string {
  return glossary[label] ?? label.replaceAll("_", " ");
}

export function buildSystemPrompt(glossary = DEFAULT_GLOSSARY): string {
  const lines = Object.entries(glossary)
    .map(([raw, text]) => `- "${raw}" = "${text}"`)
    .join("\n");
  return SYSTEM_PROMPT_TEMPLATE.replace("{glossary}", lines || "- (không có)");
}

/** Gom các chữ cái đơn liên tiếp (đánh vần) thành một nhóm. */
export function groupLabels(words: readonly string[]): Group[] {
  const groups: Group[] = [];
  for (const word of words) {
    const last = groups[groups.length - 1];
    if (isLetter(word)) {
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
      Array.isArray(item) ? `[đánh vần: ${item.map((l) => l.toUpperCase()).join("-")}]` : item,
    )
    .join(" | ");
}

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();

/** Câu dự phòng khi không gọi được LLM: dịch nhãn theo chú giải và ghép chữ cái. */
export function naiveSentence(words: readonly string[], glossary = DEFAULT_GLOSSARY): string {
  const text = groupLabels(words)
    .map((item) => (Array.isArray(item) ? capitalize(item.join("")) : labelText(item, glossary)))
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
  return (line ?? "").replace(/^["“”']+|["“”']+$/g, "");
}

export function composeMessages(words: readonly string[]) {
  return [
    { role: "system", content: buildSystemPrompt() },
    { role: "user", content: `Nhãn: ${formatLabels(words)}\nKết quả:` },
  ];
}
