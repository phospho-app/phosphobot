import { ReactNode } from 'react';
import { AuthProviderInterface, AuthState, Session } from '../types';
interface AuthContextType extends AuthState {
    login: (email: string, password: string, prefilledSession?: Session) => Promise<void>;
    signup: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    verifyEmailCode: (email: string, code: string) => Promise<void>;
    resetPassword: (email: string) => Promise<void>;
    updatePassword: (password: string) => Promise<void>;
}
export interface AuthProviderProps {
    children: ReactNode;
    provider: AuthProviderInterface;
}
export declare function AuthProvider({ children, provider }: AuthProviderProps): import("react/jsx-runtime").JSX.Element;
export declare function useAuth(): AuthContextType;
export {};
//# sourceMappingURL=AuthContext.d.ts.map