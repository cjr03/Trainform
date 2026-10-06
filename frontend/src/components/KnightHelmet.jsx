// Lucide has no knight helmet: this one matches its 24x24 stroke style.
export default function KnightHelmet(props) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M5 21v-9a7 7 0 0 1 14 0v9z" />
      <path d="M8 12h8" />
      <path d="M12 15v6" />
      <path d="M12 5c0-2 2-3 5-3" />
    </svg>
  )
}
