import { fetchWithBaseUrl } from "@/lib/utils";
import { Session } from "@/types";
import {
  ReactNode,
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

interface AuthContextType {
  session: Session | null;
  isLoading: boolean;
  proUser: boolean | null;
  login: (email: string, password: string, session?: Session) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  verifyEmailCode: (email: string, token: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(() => {
    const stored = localStorage.getItem("session");
    return stored ? JSON.parse(stored) : null;
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [proUser, setProUser] = useState<boolean | null>(() => {
    const stored = localStorage.getItem("proUser");
    return stored ? JSON.parse(stored) : null;
  });

  const login = async (
    email: string,
    password: string,
    directSession?: Session,
  ): Promise<void> => {
    if (directSession) {
      // Direct session from email confirmation
      setSession(directSession);
      localStorage.setItem("session", JSON.stringify(directSession));
      return;
    }

    const response: {
      message: string;
      session: Session;
      is_pro_user: boolean | null | undefined;
    } = await fetchWithBaseUrl("/auth/signin", "POST", {
      email,
      password,
    });
    localStorage.setItem("session", JSON.stringify(response.session));
    setSession(response.session);
    localStorage.setItem("proUser", JSON.stringify(response.is_pro_user));
    setProUser(response.is_pro_user ?? false);
  };

  const signup = async (email: string, password: string): Promise<void> => {
    const response: {
      message: string;
      session?: Session;
      is_pro_user: boolean | null | undefined;
    } = await fetchWithBaseUrl("/auth/signup", "POST", {
      email,
      password,
    });
    if (response.session) {
      localStorage.setItem("session", JSON.stringify(response.session));
      setSession(response.session);
    }
    localStorage.setItem("proUser", JSON.stringify(response.is_pro_user));
    setProUser(response.is_pro_user ?? false);
  };

  const logout = async (): Promise<void> => {
    await fetchWithBaseUrl("/auth/logout", "POST");
    localStorage.removeItem("session");
    localStorage.removeItem("proUser");
    setSession(null);
    setProUser(null);
    // Redirect to main page if we're on a protected route
    if (
      window.location.pathname === "/train" ||
      window.location.pathname === "/inference"
    ) {
      window.location.href = "/";
    }
  };

  const verifyEmailCode = async (
    email: string,
    token: string,
  ): Promise<void> => {
    const data = await fetchWithBaseUrl("/auth/verify-email-token", "POST", {
      email,
      token,
    });
    console.log("Email verification response:", data);
    if (data.session) {
      localStorage.setItem("session", JSON.stringify(data.session));
      setSession(data.session);
    } else {
      throw new Error("Invalid verification code");
    }
  };

  const validateSession = async () => {
    setIsLoading(true);
    try {
      const response: {
        authenticated: boolean;
        session: Session;
        is_pro_user: boolean;
      } = await fetchWithBaseUrl("/auth/check-auth", "GET");
      if (!response.authenticated) {
        setProUser(null);
        localStorage.removeItem("proUser");
        logout();
      }
      // Update pro user status
      setProUser(response.is_pro_user);
      localStorage.setItem("proUser", JSON.stringify(response.is_pro_user));
    } catch (e) {
      console.error("Session validation failed:", e);
      // Reset pro user status on validation failure
      setProUser(null);
      localStorage.removeItem("proUser");
      logout();
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (session) {
      validateSession();
    }
  }, [session]);

  return (
    <AuthContext.Provider
      value={{
        session,
        isLoading,
        proUser,
        login,
        signup,
        logout,
        verifyEmailCode,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
