import React, { ReactNode } from 'react';
import { useAuth } from '../context/AuthContext';

export interface ProtectedRouteProps {
    children: ReactNode;
    fallback?: ReactNode;
    requireProUser?: boolean;
}

export function ProtectedRoute({
    children,
    fallback = <div>Loading...</div>,
    requireProUser = false
}: ProtectedRouteProps) {
    const { session, isLoading, proUser } = useAuth();

    if (isLoading) {
        return <>{fallback}</>;
    }

    if (!session) {
        return <>{fallback}</>;
    }

    if (requireProUser && !proUser) {
        return <>{fallback}</>;
    }

    return <>{children}</>;
} 