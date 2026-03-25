import { Navigate, Route, Routes } from "react-router";
import { BootstrapPage } from "./pages/BootstrapPage";
import { ChatPage } from "./pages/ChatPage";
import { MainPage } from "./pages/MainPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<BootstrapPage />} />
      <Route path="/app" element={<Navigate replace to="/app/library" />} />
      <Route path="/app/library" element={<MainPage />} />
      <Route path="/app/chat" element={<ChatPage />} />
    </Routes>
  );
}
