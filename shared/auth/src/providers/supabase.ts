import { AuthProviderInterface, Session } from '../types';
import { SupabaseClient } from '@supabase/supabase-js';

export class SupabaseAuthProvider implements AuthProviderInterface {
  private supabase: SupabaseClient;

  constructor(supabaseClient: SupabaseClient) {
    this.supabase = supabaseClient;
  }

  async login(email: string, password: string): Promise<Session> {
    const { data, error } = await this.supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) throw new Error(error.message);
    if (!data.user || !data.session) throw new Error('Login failed');

    return {
      user: {
        id: data.user.id,
        email: data.user.email!,
        emailConfirmed: data.user.email_confirmed_at !== null,
      },
      accessToken: data.session.access_token,
      refreshToken: data.session.refresh_token,
      expiresIn: data.session.expires_in,
    };
  }

  async signup(email: string, password: string): Promise<void> {
    const { error } = await this.supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/confirm`,
      },
    });

    if (error) throw new Error(error.message);
  }

  async logout(): Promise<void> {
    const { error } = await this.supabase.auth.signOut();
    if (error) throw new Error(error.message);
  }

  async verifyEmailCode(email: string, code: string): Promise<Session> {
    const { data, error } = await this.supabase.auth.verifyOtp({
      email,
      token: code,
      type: 'email',
    });

    if (error) throw new Error(error.message);
    if (!data.user || !data.session) throw new Error('Verification failed');

    return {
      user: {
        id: data.user.id,
        email: data.user.email!,
        emailConfirmed: true,
      },
      accessToken: data.session.access_token,
      refreshToken: data.session.refresh_token,
      expiresIn: data.session.expires_in,
    };
  }

  async resetPassword(email: string): Promise<void> {
    const { error } = await this.supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/update-password`,
    });

    if (error) throw new Error(error.message);
  }

  async updatePassword(password: string): Promise<void> {
    const { error } = await this.supabase.auth.updateUser({ password });
    if (error) throw new Error(error.message);
  }

  async getSession(): Promise<Session | null> {
    const { data: { session } } = await this.supabase.auth.getSession();
    
    if (!session) return null;

    const { data: { user } } = await this.supabase.auth.getUser();
    if (!user) return null;

    return {
      user: {
        id: user.id,
        email: user.email!,
        emailConfirmed: user.email_confirmed_at !== null,
      },
      accessToken: session.access_token,
      refreshToken: session.refresh_token,
      expiresIn: session.expires_in,
    };
  }

  async refreshSession(): Promise<Session | null> {
    const { data: { session }, error } = await this.supabase.auth.refreshSession();
    
    if (error || !session) return null;

    const { data: { user } } = await this.supabase.auth.getUser();
    if (!user) return null;

    return {
      user: {
        id: user.id,
        email: user.email!,
        emailConfirmed: user.email_confirmed_at !== null,
      },
      accessToken: session.access_token,
      refreshToken: session.refresh_token,
      expiresIn: session.expires_in,
    };
  }

  async checkProStatus(userId: string): Promise<boolean> {
    // Check if user has an active subscription in the database
    const { data, error } = await this.supabase
      .from('subscriptions')
      .select('status')
      .eq('user_id', userId)
      .single();

    if (error || !data) return false;
    
    return data.status === 'active' || data.status === 'trialing';
  }
} 