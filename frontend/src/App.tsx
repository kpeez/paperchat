import { Routes, Route } from "react-router";
import { BootstrapPage } from "./pages/BootstrapPage";
import { MainPage } from "./pages/MainPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<BootstrapPage />} />
      <Route path="/app" element={<MainPage />} />
    </Routes>
  );
}
