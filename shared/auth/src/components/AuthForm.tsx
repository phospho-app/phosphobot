import React, { FormEvent, useState } from 'react';
import { useAuth } from '../context/AuthContext';

export interface AuthFormProps {
    mode?: 'signin' | 'signup';
    onSuccess?: () => void;
    className?: string;
}

export function AuthForm({ mode = 'signin', onSuccess, className = '' }: AuthFormProps) {
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const { login, signup } = useAuth();

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        if (!email || !password) {
            setIsLoading(false);
            return;
        }

        try {
            if (mode === 'signup') {
                await signup(email, password);
            } else {
                await login(email, password);
            }
            onSuccess?.();
        } catch (error) {
            // Error is handled in context
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit} className={`flex flex-col gap-4 ${className}`}>
            <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email"
                className="border p-2 rounded"
                required
                disabled={isLoading}
            />
            <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                className="border p-2 rounded"
                required
                disabled={isLoading}
            />
            <button
                type="submit"
                className="bg-blue-500 text-white p-2 rounded disabled:opacity-50"
                disabled={isLoading}
            >
                {isLoading ? 'Processing...' : (mode === 'signup' ? 'Sign Up' : 'Sign In')}
            </button>
        </form>
    );
} 