import { AuthForm } from "@/components/auth/AuthForm";
import { ConfirmEmail } from "@/components/auth/ConfirmEmail";
import { ForgotPassword } from "@/components/auth/ForgotPassword";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { ResetPassword } from "@/components/auth/ResetPassword";
import { Layout } from "@/components/layout/layout";
import AIControlPage from "@/pages/AIControlPage";
import AITrainingPage from "@/pages/AITrainingPage";
import AdminPage from "@/pages/AdminSettingsPage";
import BrowsePage from "@/pages/BrowsePage";
import CalibrationPage from "@/pages/CalibrationPage";
import ControlPage from "@/pages/ControlPage";
import DashboardPage from "@/pages/DashboardPage";
import GamepadControlPage from "@/pages/GamepadControlPage";
import KeyboardControlPage from "@/pages/KeyboardControlPage";
import LeaderArmControlPage from "@/pages/LeaderArmControlPage";
import NetworkPage from "@/pages/NetworkPage";
import SingleArmReplayPage from "@/pages/SingleArmReplayPage";
import ViewVideoPage from "@/pages/ViewVideoPage";
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="/control" element={<ControlPage />} />
          <Route path="/keyboard" element={<KeyboardControlPage />} />
          <Route path="/gamepad" element={<GamepadControlPage />} />
          <Route path="/leader" element={<LeaderArmControlPage />} />
          <Route path="/replay" element={<SingleArmReplayPage />} />
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/browse/:path" element={<BrowsePage />} />
          <Route
            path="/train"
            element={
              <ProtectedRoute>
                <AITrainingPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/inference"
            element={
              <ProtectedRoute>
                <AIControlPage />
              </ProtectedRoute>
            }
          />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/calibration" element={<CalibrationPage />} />
          <Route path="/network" element={<NetworkPage />} />
          <Route path="/viz" element={<ViewVideoPage />} />
          <Route path="/auth" element={<AuthForm />} />
          <Route path="/sign-in" element={<AuthForm />} />
          <Route path="/sign-up" element={<AuthForm />} />
          <Route path="/auth/confirm" element={<ConfirmEmail />} />
          <Route path="/auth/forgot-password" element={<ForgotPassword />} />
          <Route path="/auth/reset-password" element={<ResetPassword />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
