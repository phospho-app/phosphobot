"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DashboardAuthProvider = void 0;
class DashboardAuthProvider {
    constructor(baseUrl) {
        this.baseUrl = baseUrl || '';
    }
    async fetchWithBaseUrl(endpoint, method, body) {
        const url = this.baseUrl ? `${this.baseUrl}${endpoint}` : endpoint;
        const response = await fetch(url, {
            method: method || 'GET',
            headers: { 'Content-Type': 'application/json' },
            body: body ? JSON.stringify(body) : null,
        });
        if (response.ok)
            return await response.json();
        let errorMessage = response.statusText;
        try {
            const errorData = await response.json();
            if (errorData.detail)
                errorMessage = errorData.detail;
            if (typeof errorMessage === 'object') {
                errorMessage = JSON.stringify(errorMessage);
            }
        }
        catch {
            // Fallback to statusText if JSON parsing fails
        }
        throw new Error(errorMessage);
    }
    async login(email, password) {
        const data = await this.fetchWithBaseUrl('/auth/signin', 'POST', {
            email,
            password,
        });
        if (data.session) {
            const session = {
                user: {
                    id: data.session.user_id,
                    email: data.session.user_email,
                    emailConfirmed: data.session.email_confirmed,
                },
                accessToken: data.session.access_token,
                refreshToken: data.session.refresh_token,
                expiresIn: data.session.expires_in,
            };
            localStorage.setItem('session', JSON.stringify(session));
            return session;
        }
        throw new Error('Login failed');
    }
    async signup(email, password) {
        await this.fetchWithBaseUrl('/auth/signup', 'POST', {
            email,
            password,
        });
    }
    async logout() {
        const sessionStr = localStorage.getItem('session');
        if (sessionStr) {
            const session = JSON.parse(sessionStr);
            await this.fetchWithBaseUrl('/auth/signout', 'POST', {
                refresh_token: session.refreshToken,
            });
        }
        localStorage.removeItem('session');
    }
    async verifyEmailCode(email, code) {
        const data = await this.fetchWithBaseUrl('/auth/verify', 'POST', {
            email,
            code,
        });
        if (data.session) {
            const session = {
                user: {
                    id: data.session.user_id,
                    email: data.session.user_email,
                    emailConfirmed: data.session.email_confirmed,
                },
                accessToken: data.session.access_token,
                refreshToken: data.session.refresh_token,
                expiresIn: data.session.expires_in,
            };
            localStorage.setItem('session', JSON.stringify(session));
            return session;
        }
        throw new Error('Verification failed');
    }
    async resetPassword(email) {
        await this.fetchWithBaseUrl('/auth/forgot-password', 'POST', {
            email,
        });
    }
    async updatePassword(password) {
        const sessionStr = localStorage.getItem('session');
        if (!sessionStr)
            throw new Error('No session found');
        const session = JSON.parse(sessionStr);
        await this.fetchWithBaseUrl('/auth/update-password', 'POST', {
            password,
            access_token: session.accessToken,
        });
    }
    async getSession() {
        const sessionStr = localStorage.getItem('session');
        if (!sessionStr)
            return null;
        try {
            const sessionData = JSON.parse(sessionStr);
            return {
                user: {
                    id: sessionData.user_id || sessionData.user?.id,
                    email: sessionData.user_email || sessionData.user?.email,
                    emailConfirmed: sessionData.email_confirmed || sessionData.user?.emailConfirmed,
                },
                accessToken: sessionData.access_token || sessionData.accessToken,
                refreshToken: sessionData.refresh_token || sessionData.refreshToken,
                expiresIn: sessionData.expires_in || sessionData.expiresIn,
            };
        }
        catch {
            return null;
        }
    }
    async refreshSession() {
        const session = await this.getSession();
        if (!session)
            return null;
        try {
            const data = await this.fetchWithBaseUrl('/auth/refresh', 'POST', {
                refresh_token: session.refreshToken,
            });
            if (data.session) {
                const newSession = {
                    user: {
                        id: data.session.user_id,
                        email: data.session.user_email,
                        emailConfirmed: data.session.email_confirmed,
                    },
                    accessToken: data.session.access_token,
                    refreshToken: data.session.refresh_token,
                    expiresIn: data.session.expires_in,
                };
                localStorage.setItem('session', JSON.stringify(newSession));
                return newSession;
            }
        }
        catch {
            // If refresh fails, clear the session
            localStorage.removeItem('session');
        }
        return null;
    }
    async checkProStatus(userId) {
        try {
            const data = await this.fetchWithBaseUrl('/auth/pro-status', 'POST', {
                user_id: userId,
            });
            return data.is_pro || false;
        }
        catch {
            return false;
        }
    }
}
exports.DashboardAuthProvider = DashboardAuthProvider;
