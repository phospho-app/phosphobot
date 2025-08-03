import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types
interface SignupRequest {
  email: string;
  password: string;
}

// Helper function to get local IP (simplified version)
function getLocalUrl(): string {
  const port = process.env.PORT || "3000";
  // In production, you'd want to use your actual domain
  // For local development, you can use localhost
  return process.env.NODE_ENV === "production"
    ? `https://${process.env.DOMAIN}`
    : `http://localhost:${port}`;
}

export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const body: SignupRequest = await request.json();
    const { email, password } = body;

    // Log the user email
    console.log("Sign up attempt for email:", email);

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      );
    }

    // Sign up the user with Supabase
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${getLocalUrl()}/`,
      },
    });

    if (error) {
      console.error("Supabase signup error:", error);
      return NextResponse.json(
        { error: `Signup failed: ${error.message}` },
        { status: 500 }
      );
    }

    if (!data.user) {
      return NextResponse.json({ error: "Signup failed" }, { status: 400 });
    }

    console.log("Signup successful for email:", email);

    return NextResponse.json(
      { message: "Sign up successful", email },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error in signup route:", error);
    return NextResponse.json(
      { error: "Failed to process sign up request" },
      { status: 400 }
    );
  }
}
