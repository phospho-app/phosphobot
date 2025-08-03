'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import { createClient } from '@/lib/supabase/client';
import { useRouter } from 'next/navigation';
import Header from '@/components/sections/header';
import HeroSection from '@/components/sections/hero';
import ComparisonTable from '@/components/sections/comparison-table';
import PricingCard from '@/components/sections/pricing-card';
import FAQSection from '@/components/sections/faq';
import SubscribeButton from '@/components/sections/subscribe-button';

function HomeContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const urlEmail = searchParams.get('user_email') || null;
    const [userEmail, setUserEmail] = useState<string | null>(urlEmail);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        const checkUser = async () => {
            const supabase = createClient();
            const { data: { user } } = await supabase.auth.getUser();

            // If user is logged in, use their email, otherwise keep URL parameter
            if (user?.email) {
                setUserEmail(user.email);
                setIsAuthenticated(true);
            }
        };

        checkUser();
    }, []);

    const handleLogout = async () => {
        const supabase = createClient();
        await supabase.auth.signOut();
        setUserEmail(urlEmail);
        setIsAuthenticated(false);
        router.refresh();
    };

    // if there is to_checkout=true in the URL, redirect to the checkout page
    useEffect(() => {
        if (searchParams.get('to_checkout') === 'true' && userEmail) {
            // Create and submit a form to start checkout session
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = '/api/checkout_sessions';

            const emailInput = document.createElement('input');
            emailInput.type = 'hidden';
            emailInput.name = 'user_email';
            emailInput.value = userEmail;

            form.appendChild(emailInput);
            document.body.appendChild(form);
            form.submit();
        }
    }, [searchParams, userEmail]);

    return (
        <div className="bg-gray-background min-h-screen font-sans">
            <Header />
            <main className="container mx-auto px-4 max-w-7xl">
                <HeroSection />
                <section className="py-16">
                    <div className="max-w-6xl mx-auto flex flex-col lg:flex-row items-start lg:space-x-8">
                        <div className="lg:flex-1 w-full lg:w-auto">
                            <ComparisonTable />
                        </div>
                        <div className="lg:w-96 w-full mt-8 lg:mt-0">
                            <PricingCard userEmail={userEmail} />
                        </div>
                    </div>
                </section>
                <section className="py-4">
                    <h2 className="text-4xl font-headline text-dark-gray mb-4 text-center lg:text-left max-w-4xl mx-auto px-8">
                        Frequently Asked Questions
                    </h2>
                    <FAQSection />
                </section>
                <section className="py-4 mb-6">
                    <SubscribeButton userEmail={userEmail} />
                </section>
                <section className="mb-4">
                    <div className="flex justify-center">
                        {isAuthenticated ? (
                            <button
                                onClick={handleLogout}
                                className="text-gray-500 hover:text-gray-700 text-sm underline transition-colors"
                            >
                                Logout
                            </button>
                        ) : (
                            '💚 from Paris'
                        )}
                    </div>
                </section>
            </main>
        </div>
    );
}

export default function Home() {
    return (
        <Suspense fallback={null}>
            <HomeContent />
        </Suspense>
    );
}