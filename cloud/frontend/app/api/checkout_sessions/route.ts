import { NextResponse } from 'next/server'
import { headers } from 'next/headers'

import { stripe } from '../../../lib/stripe'
import { createClient } from '../../../lib/supabase/server'

// We use se process.env.STRIPE_MODE to deteremine if we are in TEST or PROD

export async function POST() {
  try {
    const headersList = await headers()
    const origin = headersList.get('origin')

    // The checkout is always for the signed-in user, read from the Supabase
    // session cookie. Never trust a user id or email sent by the client.
    const supabase = await createClient()
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // The Stripe webhook needs the email to link the subscription to the user
    if (!user.email) {
      return NextResponse.json({ error: 'Your account has no email address' }, { status: 400 })
    }

    // Create a Checkout Session for the signed-in user.
    const session = await stripe.checkout.sessions.create({
      line_items: [
        {
          price: process.env.STRIPE_MODE === 'TEST' ? 'price_1RgkS6KMbS7I1rNchpmAyKGD' : 'price_1RTmduKMbS7I1rNcnMAIEFth',
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: 'https://a6gysudbx4d.typeform.com/to/CFFeM0Nj',
      cancel_url: `${origin}/?canceled=true`,
      automatic_tax: {enabled: true},
      // Prefill the user email
      customer_email: user.email,
      allow_promotion_codes: true,
      // Add metadata to the checkout session
      metadata: {
        supabase_user_id: user.id,
        supabase_user_email: user.email
      },
      // Add metadata to the subscription
      subscription_data: {
        metadata: {
          supabase_user_id: user.id,
          supabase_user_email: user.email
        }
      }
    });

    if (!session.url) {
      throw new Error('Failed to create checkout session URL')
    }

    console.log('Created a checkout session for user', user.id)

    return NextResponse.redirect(session.url, 303)
  } catch (err) {
    const error = err as Error & { statusCode?: number }
    return NextResponse.json(
      { error: error.message || 'An error occurred' },
      { status: error.statusCode || 500 }
    )
  }
}
