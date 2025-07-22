export interface User {
  id: string;
  email: string;
  emailConfirmed?: boolean;
}

export interface Session {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn?: number;
}

export interface AuthState {
  session: Session | null;
  isLoading: boolean;
  proUser: boolean;
  error: string | null;
}

export interface AuthProviderInterface {
  login(email: string, password: string): Promise<Session>;
  signup(email: string, password: string): Promise<void>;
  logout(): Promise<void>;
  verifyEmailCode(email: string, code: string): Promise<Session>;
  resetPassword(email: string): Promise<void>;
  updatePassword(password: string): Promise<void>;
  getSession(): Promise<Session | null>;
  refreshSession(): Promise<Session | null>;
  checkProStatus(userId: string): Promise<boolean>;
} 