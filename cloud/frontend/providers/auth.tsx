'use client';

import { ReactNode } from 'react';
import { AuthProvider as SharedAuthProvider, SupabaseAuthProvider } from '@phosphobot/shared-auth';
import { createClient } from '@/lib/supabase/client';

const supabaseClient = createClient();
const supabaseAuthProvider = new SupabaseAuthProvider(supabaseClient);

export function AuthProvider({ children }: { children: ReactNode }) {
    return (
        <SharedAuthProvider provider={supabaseAuthProvider}>
            {children}
        </SharedAuthProvider>
    );
} 