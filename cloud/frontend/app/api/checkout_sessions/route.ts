import { NextResponse } from 'next/server'
import { headers } from 'next/headers'

import { stripe } from '../../../lib/stripe'
import { createClient } from '@supabase/supabase-js'

// We use se process.env.STRIPE_MODE to deteremine if we are in TEST or PROD

export async function POST(request: Request) {
  try {
    const headersList = await headers()
    const origin = headersList.get('origin')

    // Parse form data to get userId and userEmail
    const formData = await request.formData()
    const userEmail = formData.get('user_email') as string
    // The frontend doesn't pass the userId anymore
    // So using the user email to get the userId from Supabase
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
    // Find user by email with pagination support
    let userId = null;
    let page = 1;
    let hasMore = true;
    
    while (hasMore && !userId) {
      const { data, error } = await supabase.auth.admin.listUsers({
        page,
        perPage: 50
      })
      
      if (error) {
        console.error(error);
        return NextResponse.json({ error: error.message }, { status: 500 });
      }
      
      if (data?.users) {
        const user = data.users.find(u => u.email === userEmail);
        if (user) {
          userId = user.id;
        }
        
        // Check if there are more pages
        hasMore = data.users.length === 50;
        page++;
      } else {
        hasMore = false;
      }
    }

    if (!userId) {
      console.error('User not found');
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Create Checkout Sessions from body params.
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
      customer_email: userEmail,
      allow_promotion_codes: true,
      // Add metadata to the checkout session
      metadata: {
        supabase_user_id: userId,
        supabase_user_email: userEmail
      },
      // Add metadata to the subscription
      subscription_data: {
        metadata: {
          supabase_user_id: userId,
          supabase_user_email: userEmail
        }
      }
    });
    
    if (!session.url) {
      throw new Error('Failed to create checkout session URL')
    }

    console.log('Created a checkout session for user', userId, 'with email', userEmail)
    
    return NextResponse.redirect(session.url, 303)
  } catch (err) {
    const error = err as Error & { statusCode?: number }
    return NextResponse.json(
      { error: error.message || 'An error occurred' },
      { status: error.statusCode || 500 }
    )
  }
}