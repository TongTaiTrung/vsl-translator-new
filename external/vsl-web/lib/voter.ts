/** Word voting (port of src/app/voting.py). */

export class WordVoter {
  sequence: string[] = [];
  candidate: string | null = null;
  count = 0;
  blocked: string | null = null;

  constructor(
    public minConsecutive = 2,
    public maxWords = 64,
  ) {}

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
