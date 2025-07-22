# Shared Authentication Module

This module provides a unified authentication system for phosphobot applications, supporting both custom backend API (dashboard) and Supabase (cloud frontend).

## Features

- Unified authentication interface for multiple providers
- Support for dashboard's custom backend API
- Support for Supabase authentication
- Session management with automatic refresh
- Pro user status tracking
- Protected routes
- Email verification support

## Installation

```bash
npm install @phosphobot/shared-auth
```

## Usage

### Dashboard App (Custom Backend)

```tsx
// In src/providers/auth.tsx
import {
  AuthProvider as SharedAuthProvider,
  DashboardAuthProvider,
} from "@phosphobot/shared-auth";

const dashboardAuthProvider = new DashboardAuthProvider();

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <SharedAuthProvider provider={dashboardAuthProvider}>
      {children}
    </SharedAuthProvider>
  );
}
```

### Cloud Frontend (Supabase)

```tsx
// In providers/auth.tsx
import {
  AuthProvider as SharedAuthProvider,
  SupabaseAuthProvider,
} from "@phosphobot/shared-auth";
import { createClient } from "@/lib/supabase/client";

const supabaseClient = createClient();
const supabaseAuthProvider = new SupabaseAuthProvider(supabaseClient);

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <SharedAuthProvider provider={supabaseAuthProvider}>
      {children}
    </SharedAuthProvider>
  );
}
```

### Using Auth in Components

```tsx
import { useAuth } from "@phosphobot/shared-auth";

function MyComponent() {
  const { session, isLoading, proUser, login, logout } = useAuth();

  if (isLoading) return <div>Loading...</div>;

  if (!session) {
    return <button onClick={() => login("email", "password")}>Login</button>;
  }

  return (
    <div>
      <p>Welcome, {session.user.email}!</p>
      {proUser && <p>You are a Pro user!</p>}
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

### Protected Routes

```tsx
import { ProtectedRoute } from '@phosphobot/shared-auth';

<ProtectedRoute>
  <MyProtectedComponent />
</ProtectedRoute>

// With custom fallback
<ProtectedRoute fallback={<Navigate to="/login" />}>
  <MyProtectedComponent />
</ProtectedRoute>

// Require Pro user
<ProtectedRoute requireProUser={true}>
  <ProOnlyFeature />
</ProtectedRoute>
```

## Development

```bash
# Install dependencies
npm install

# Build the module
npm run build

# Watch for changes
npm run dev
```
