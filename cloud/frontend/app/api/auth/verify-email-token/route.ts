import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

interface VerifyEmailCodeRequest {
  email: string;
  token: string;
}

export async function POST(request: NextRequest) {
    try {
      const body: VerifyEmailCodeRequest = await request.json();
      const { email, token } = body;
  
      // Initialize your Supabase client (adjust based on your setup)
      const supabase = createClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.SUPABASE_SERVICE_ROLE_KEY!
      );
  
      const { data, error } = await supabase.auth.verifyOtp({
        email,
        token,
        type: 'email',
      });
  
      if (error || !data.user || !data.session || !data.user.email) {
        console.error('Error verifying email token:', error);
        return NextResponse.json(
          { error: 'Invalid or expired email verification token.' },
          { status: 400 }
        );
      }
  
      // Return success response
      return NextResponse.json({
        user: data.user,
        session: data.session,
      });
  
    } catch {
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      );
    }
  }