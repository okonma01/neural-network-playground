function Tooltip({ label, text }) {
  return (
    <span className="tooltip-inline" tabIndex={0}>
      {label}
      <span className="tooltip-bubble" role="tooltip">
        {text}
      </span>
    </span>
  );
}

export default Tooltip;