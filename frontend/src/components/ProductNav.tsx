import { NavLink } from "react-router";

export function ProductNav() {
  return (
    <nav className="flex flex-wrap items-center gap-1 rounded-full border border-slate-200 p-1">
      <NavItem label="Library" to="/app/library" />
      <NavItem label="Chat" to="/app/chat" />
    </nav>
  );
}

function NavItem({ label, to }: { label: string; to: string }) {
  return (
    <NavLink
      className={({ isActive }) =>
        `rounded-full px-3 py-1.5 text-sm font-medium transition ${
          isActive
            ? "bg-slate-950 text-white"
            : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
        }`
      }
      to={to}
    >
      {label}
    </NavLink>
  );
}
