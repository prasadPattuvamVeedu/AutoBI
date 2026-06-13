import { Navigate, Route, Routes } from "react-router-dom";

import Navbar from "./components/Navbar.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";
import { AuthProvider } from "./context/AuthContext.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import DatasetDetailPage from "./pages/DatasetDetailPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import UploadDatasetPage from "./pages/UploadDatasetPage.jsx";


function App() {
  return (
    <AuthProvider>
      <Navbar />
      <main className="app-shell">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/datasets/upload"
            element={
              <ProtectedRoute>
                <UploadDatasetPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/datasets/:id"
            element={
              <ProtectedRoute>
                <DatasetDetailPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </main>
    </AuthProvider>
  );
}

export default App;
