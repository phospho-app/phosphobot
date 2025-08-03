# phospho

## Features

Subscription management with Stripe for PRO users

## Installation

Add a `.env.local` file and update the following variables:

```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=
SUPABASE_SERVICE_ROLE_KEY=
STRIPE_SECRET_KEY=
STRIPE_MODE=TEST
```

Both `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` can be found in [your Supabase project's API settings](https://supabase.com/dashboard/project/_?showConnect=true)

`NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` and `STRIPE_SECRET_KEY` can be found in [your Stripe dashboard](https://dashboard.stripe.com/test/apikeys)

`STRIPE_MODE` can be set to `TEST` or `PROD`, depending on whether you want to use the test or production environment of Stripe.

You can now run the Next.js local development server:

```bash
npm run dev
```

The app should now be running on [localhost:3000](http://localhost:3000/).

## Passing the user email

You can pass it as an URL parameter to the page:

```
http://localhost:3000/?user_email=test%40example.com
```

The @ symbol is encoded as %40 in URLs.
