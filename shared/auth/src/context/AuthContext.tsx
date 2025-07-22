import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { AuthProviderInterface, AuthState, Session } from '../types';

interface AuthContextType extends AuthState {
    login: (email: string, password: string, prefilledSession?: Session) => Promise<void>;
    signup: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    verifyEmailCode: (email: string, code: string) => Promise<void>;
    resetPassword: (email: string) => Promise<void>;
    updatePassword: (password: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export interface AuthProviderProps {
    children: ReactNode;
    provider: AuthProviderInterface;
}

export function AuthProvider({ children, provider }: AuthProviderProps) {
    const [state, setState] = useState<AuthState>({
        session: null,
        isLoading: true,
        proUser: false,
        error: null,
    });

    useEffect(() => {
        const initAuth = async () => {
            try {
                const session = await provider.getSession();
                if (session) {
                    const proUser = await provider.checkProStatus(session.user.id);
                    setState({
                        session,
                        isLoading: false,
                        proUser,
                        error: null,
                    });
                } else {
                    setState({
                        session: null,
                        isLoading: false,
                        proUser: false,
                        error: null,
                    });
                }
            } catch (error) {
                setState({
                    session: null,
                    isLoading: false,
                    proUser: false,
                    error: error instanceof Error ? error.message : 'Failed to initialize auth',
                });
            }
        };

        initAuth();

        // Set up session refresh interval
        const refreshInterval = setInterval(async () => {
            if (state.session) {
                try {
                    const newSession = await provider.refreshSession();
                    if (newSession) {
                        setState(prev => ({ ...prev, session: newSession }));
                    } else {
                        setState({
                            session: null,
                            isLoading: false,
                            proUser: false,
                            error: null,
                        });
                    }
                } catch {
                    // Silent fail on refresh
                }
            }
        }, 30 * 60 * 1000); // Refresh every 30 minutes

        return () => clearInterval(refreshInterval);
    }, [provider]);

    const login = async (email: string, password: string, prefilledSession?: Session) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            const session = prefilledSession || await provider.login(email, password);
            const proUser = await provider.checkProStatus(session.user.id);
            setState({
                session,
                isLoading: false,
                proUser,
                error: null,
            });
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Login failed',
            }));
            throw error;
        }
    };

    const signup = async (email: string, password: string) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            await provider.signup(email, password);
            setState(prev => ({ ...prev, isLoading: false }));
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Signup failed',
            }));
            throw error;
        }
    };

    const logout = async () => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            await provider.logout();
            setState({
                session: null,
                isLoading: false,
                proUser: false,
                error: null,
            });
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Logout failed',
            }));
            throw error;
        }
    };

    const verifyEmailCode = async (email: string, code: string) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            const session = await provider.verifyEmailCode(email, code);
            const proUser = await provider.checkProStatus(session.user.id);
            setState({
                session,
                isLoading: false,
                proUser,
                error: null,
            });
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Verification failed',
            }));
            throw error;
        }
    };

    const resetPassword = async (email: string) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            await provider.resetPassword(email);
            setState(prev => ({ ...prev, isLoading: false }));
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Reset password failed',
            }));
            throw error;
        }
    };

    const updatePassword = async (password: string) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        try {
            await provider.updatePassword(password);
            setState(prev => ({ ...prev, isLoading: false }));
        } catch (error) {
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: error instanceof Error ? error.message : 'Update password failed',
            }));
            throw error;
        }
    };

    return (
        <AuthContext.Provider
            value={{
                ...state,
                login,
                signup,
                logout,
                verifyEmailCode,
                resetPassword,
                updatePassword,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
} 