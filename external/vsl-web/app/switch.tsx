
export function Switch(props: {
  label: string;
  hint?: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  const { label, hint, checked, disabled } = props;
  return (
    <label className={`switch-row${disabled ? " disabled" : ""}`}>
      <span className="switch-text">
        <span>{label}</span>
        {hint && <span className="help">{hint}</span>}
      </span>
      <input
        type="checkbox" role="switch" checked={checked} disabled={disabled}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      <span className="switch" aria-hidden="true"><span className="switch-thumb" /></span>
    </label>
  );
}
