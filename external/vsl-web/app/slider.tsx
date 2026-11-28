/** Range slider control. */

export function Slider(props: {
  min: number;
  max: number;
  step: number;
  value: number;
  disabled?: boolean;
  stops?: boolean;
  ariaLabel?: string;
  valueText?: string;
  onChange: (value: number) => void;
}) {
  const { min, max, step, value, disabled, stops } = props;
  const p = max > min ? (value - min) / (max - min) : 0;
  const count = stops ? Math.round((max - min) / step) + 1 : 0;
  return (
    <div
      className={`slider${stops ? " stepped" : ""}${disabled ? " disabled" : ""}`}
      style={{ "--p": p } as React.CSSProperties}
    >
      <span className="slider-rail" aria-hidden="true"><span className="slider-fill" /></span>
      {Array.from({ length: count }, (_, i) => (
        <span
          key={i} aria-hidden="true"
          className={`slider-stop${i / (count - 1) <= p + 1e-9 ? " reached" : ""}`}
          style={{ "--s": i / (count - 1) } as React.CSSProperties}
        />
      ))}
      <span className="slider-thumb" aria-hidden="true" />
      <input
        className="slider-input" type="range"
        min={min} max={max} step={step} value={value} disabled={disabled}
        aria-label={props.ariaLabel} aria-valuetext={props.valueText}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </div>
  );
}
