// Bản TypeScript của src/app/voting.py (WordVoter): một nhãn phải thắng N phiếu liên tiếp mới được chốt.
// Ở web, mỗi phản hồi của /api/predict cho một cửa sổ 30 frame là một phiếu.

export class WordVoter {
  sequence: string[] = [];
  candidate: string | null = null;
  count = 0;
  /** Từ vừa chốt: không chốt lại cho tới khi người dùng nghỉ tay hoặc chốt một từ khác. */
  blocked: string | null = null;

  constructor(
    public minConsecutive = 2,
    public maxWords = 64,
  ) {}

  /** Ghi nhận một phiếu; trả về nhãn nếu nó vừa được thêm vào chuỗi. */
  vote(label: string | null): string | null {
    if (label === null) {
      this.interrupt();
      return null;
    }
    if (label === this.candidate) {
      this.count += 1;
    } else {
      this.candidate = label;
      this.count = 1;
    }
    if (label === this.blocked || this.count < this.minConsecutive) return null;
    this.sequence.push(label);
    if (this.sequence.length > this.maxWords) this.sequence.splice(0, this.sequence.length - this.maxWords);
    this.blocked = label;
    this.interrupt();
    return label;
  }

  interrupt(): void {
    this.candidate = null;
    this.count = 0;
  }

  /** Tay đã rời khung: đứt chuỗi đếm và cho phép lặp lại từ vừa chốt. */
  release(): void {
    this.interrupt();
    this.blocked = null;
  }

  undo(): string | null {
    this.release();
    return this.sequence.pop() ?? null;
  }

  clear(): void {
    this.release();
    this.sequence = [];
  }
}
