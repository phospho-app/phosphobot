'use client';

import { ReactNode } from 'react';
import { AuthProvider } from '@/providers/auth';

export function Providers({ children }: { children: ReactNode }) {
    return (
        <AuthProvider>
            {children}
        </AuthProvider>
    );
} 