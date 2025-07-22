import { ReactNode } from 'react';
export interface ProtectedRouteProps {
    children: ReactNode;
    fallback?: ReactNode;
    requireProUser?: boolean;
}
export declare function ProtectedRoute({ children, fallback, requireProUser }: ProtectedRouteProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=ProtectedRoute.d.ts.map