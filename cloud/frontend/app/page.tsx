'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import { useAuth } from '@phosphobot/shared-auth';
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
    const { session, logout } = useAuth();

    useEffect(() => {
        // If user is logged in, use their email, otherwise keep URL parameter
        if (session?.user?.email) {
            setUserEmail(session.user.email);
        }
    }, [session]);

    const handleLogout = async () => {
        await logout();
        setUserEmail(urlEmail);
        router.refresh();
    };

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
                        {session ? (
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