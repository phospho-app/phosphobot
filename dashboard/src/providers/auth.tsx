import { ReactNode } from 'react';
import { AuthProvider as SharedAuthProvider, DashboardAuthProvider } from '@phosphobot/shared-auth';

const dashboardAuthProvider = new DashboardAuthProvider();

export function AuthProvider({ children }: { children: ReactNode }) {
    return (
        <SharedAuthProvider provider={dashboardAuthProvider}>
            {children}
        </SharedAuthProvider>
    );
} 