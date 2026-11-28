/** Personal sign memory UI: record, list and delete user exemplars. */

"use client";

import { useEffect, useRef, useState } from "react";
import type { CommittedWindow } from "@/lib/recognizer";
import { MAX_AUTO_PER_LABEL, MAX_PER_LABEL, type Sample, sourceOf } from "@/lib/memory";
import { Slider } from "./slider";
import { Switch } from "./switch";

interface Props {
  open: boolean;
  onClose: () => void;
  classes: string[];
  samples: Sample[];
  staleCount: number;
  enabled: boolean;
  strength: number;
  autoLearn: boolean;
  onAutoLearn: (value: boolean) => void;
  learnMode: "realtime" | "sentence";
  onLearnMode: (value: "realtime" | "sentence") => void;
  running: boolean;
  busy: boolean;
  activeExemplars: number | null;
  memoryError: string | null;
  committed: CommittedWindow | null;
  onEnabled: (value: boolean) => void;
  onStrength: (value: number) => void;
  onRecord: (label: string) => void;
  onSaveCorrection: (label: string) => void;
  onDeleteLabel: (label: string) => void;
  onClearAll: () => void;
}

export function PersonalizeSheet(props: Props) {
  const { open, onClose, classes, samples, enabled, strength, autoLearn, learnMode, running, busy, committed } = props;
  const sheetRef = useRef<HTMLDivElement>(null);
  const [correction, setCorrection] = useState("");

  useEffect(() => {
    if (!open) return;
    sheetRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => setCorrection(committed?.word ?? ""), [committed]);

  const counts = new Map<string, { manual: number; auto: number }>();
  samples.forEach((s) => {
    const c = counts.get(s.label) ?? { manual: 0, auto: 0 };
    c[sourceOf(s)] += 1;
    counts.set(s.label, c);
  });
  const autoTotal = samples.filter((s) => sourceOf(s) === "auto").length;
  const recordable = running && !busy;

  const status = !enabled
    ? "Đang tắt: dùng mô hình gốc."
    : samples.length === 0
      ? "Chưa có mẫu nào."
      : props.activeExemplars != null
        ? `Đang dùng ${props.activeExemplars} mẫu.`
        : running
          ? "Mẫu sẽ áp dụng từ lần dự đoán kế tiếp."
          : `${samples.length} mẫu sẽ được dùng khi bật camera.`;

  return (
    <>
      <div className={`sheet-backdrop ${open ? "open" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside
        className={`sheet ${open ? "open" : ""}`}
        aria-label="Cá nhân hoá"
        aria-hidden={!open}
        inert={!open}
        ref={sheetRef}
        tabIndex={-1}
      >
        <div className="sheet-head">
          <h2>Cá nhân hoá</h2>
          <button className="icon-button" onClick={onClose} aria-label="Đóng cá nhân hoá">
            <svg className="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"
              strokeWidth="1.8" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>

        <div className="sheet-body">
          <section className="group">
            <p className="help">
              Ký vài từ theo cách của bạn để mô hình nhận ra bạn tốt hơn, không cần train lại. Mẫu (chỉ landmark,
              không có hình) lưu trên trình duyệt này và chỉ gửi lên server khi nhận diện.
            </p>
            <Switch label="Dùng mẫu của tôi khi nhận diện" checked={enabled} onChange={props.onEnabled} />
            <label className="field">
              <span>Mức ảnh hưởng: {Math.round(strength * 100)}%</span>
              <Slider
                min={0.1} max={1} step={0.05} value={strength} disabled={!enabled}
                valueText={`${Math.round(strength * 100)}%`} onChange={props.onStrength}
              />
            </label>
            <Switch
              label="Tự học từ các câu tôi ký" checked={autoLearn} disabled={!enabled} onChange={props.onAutoLearn}
            />
            <div className="memory-mode" role="radiogroup" aria-label="Khi nào tự học">
              <label className="check">
                <input type="radio" name="learn-mode" checked={learnMode === "realtime"} disabled={!enabled || !autoLearn}
                  onChange={() => props.onLearnMode("realtime")} />
                <span>Học ngay sau mỗi từ (thời gian thực)</span>
              </label>
              <label className="check">
                <input type="radio" name="learn-mode" checked={learnMode === "sentence"} disabled={!enabled || !autoLearn}
                  onChange={() => props.onLearnMode("sentence")} />
                <span>Học khi kết thúc câu</span>
              </label>
            </div>
            <p className="help">
              {learnMode === "realtime"
                ? "Từ mô hình khá chắc được lưu làm mẫu vài giây sau khi chốt và dùng ngay cho từ kế tiếp. Xoá hoặc hoàn tác từ đó sẽ xoá mẫu đã học."
                : "Khi hết câu, những từ bạn giữ lại mà mô hình khá chắc sẽ được lưu làm mẫu."}
              {` Tối đa ${MAX_AUTO_PER_LABEL} mẫu tự học mỗi từ. Từ nhận sai: hoàn tác hoặc dùng “Sửa từ vừa nhận”.`}
            </p>
            <p className="help">
              {status}
              {autoTotal > 0 && ` Trong đó ${autoTotal} mẫu tự học.`}
            </p>
            {props.memoryError && enabled && (
              <p className="error" role="alert">Server từ chối mẫu, đang dùng mô hình gốc: {props.memoryError}</p>
            )}
            {props.staleCount > 0 && (
              <p className="help">{props.staleCount} mẫu thuộc phiên bản đặc trưng cũ, không được dùng. Hãy ghi lại.</p>
            )}
          </section>

          <section className="group">
            <h3>Sửa từ vừa nhận</h3>
            {committed ? (
              <>
                <p className="help">
                  Vừa chốt <strong>{committed.word}</strong>. Nếu sai, chọn từ đúng rồi lưu làm mẫu.
                </p>
                <div className="memory-correct">
                  <select value={correction} onChange={(e) => setCorrection(e.target.value)} aria-label="Từ đúng">
                    {classes.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <button className="btn btn-ghost" disabled={busy || !correction}
                    onClick={() => props.onSaveCorrection(correction)}>
                    Lưu làm mẫu
                  </button>
                </div>
              </>
            ) : (
              <p className="help">Từ vừa chốt sẽ hiện ở đây để bạn sửa và lưu làm mẫu.</p>
            )}
          </section>

          <section className="group">
            <h3>Mẫu theo từ</h3>
            {!running && <p className="help">Bật camera để ghi mẫu.</p>}
            <ul className="memory-list">
              {classes.map((label) => {
                const c = counts.get(label) ?? { manual: 0, auto: 0 };
                const n = c.manual + c.auto;
                return (
                  <li key={label}>
                    <span className="memory-label">{label}</span>
                    <span className="memory-dots" aria-label={`${c.manual} mẫu ghi tay, ${c.auto} mẫu tự học, tối đa ${MAX_PER_LABEL}`}
                      title={`${c.manual} ghi tay · ${c.auto} tự học`}>
                      {Array.from({ length: MAX_PER_LABEL }, (_, i) => (
                        <i key={i} className={i < c.manual ? "on" : i < n ? "auto" : ""} />
                      ))}
                    </span>
                    <button className="link" disabled={!recordable} onClick={() => props.onRecord(label)}>
                      Ghi mẫu
                    </button>
                    <button className="link danger" disabled={n === 0 || busy} onClick={() => props.onDeleteLabel(label)}
                      aria-label={`Xoá mẫu của ${label}`}>
                      Xoá
                    </button>
                  </li>
                );
              })}
            </ul>
            <p className="help">
              Tối đa {MAX_PER_LABEL} mẫu mỗi từ (chấm đặc: ghi tay, chấm viền: tự học). Mẫu mới thay mẫu cũ nhất;
              mẫu tự học không thay mẫu ghi tay.
            </p>
          </section>

          <button
            className="link danger"
            disabled={samples.length === 0 || busy}
            onClick={() => {
              if (window.confirm(`Xoá ${samples.length} mẫu cá nhân trên trình duyệt này? Không hoàn tác được.`)) {
                props.onClearAll();
              }
            }}
          >
            Xoá toàn bộ dữ liệu cá nhân
          </button>
        </div>
      </aside>
    </>
  );
}
