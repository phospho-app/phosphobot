import { AuthProviderInterface, Session } from '../types';
export declare class DashboardAuthProvider implements AuthProviderInterface {
    private baseUrl;
    constructor(baseUrl?: string);
    private fetchWithBaseUrl;
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
//# sourceMappingURL=dashboard.d.ts.map