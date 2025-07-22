import { AuthProviderInterface, Session } from '../types';
import { SupabaseClient } from '@supabase/supabase-js';
export declare class SupabaseAuthProvider implements AuthProviderInterface {
    private supabase;
    constructor(supabaseClient: SupabaseClient);
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
//# sourceMappingURL=supabase.d.ts.map