"use client";

import { useRouter } from 'next/navigation';

interface SubscribeButtonProps {
    className?: string;
    children?: React.ReactNode;
    onClick?: () => void;
    userEmail?: string | null;
}

export default function SubscribeButton({
    className = "",
    children = "Subscribe",
    onClick,
    userEmail
}: SubscribeButtonProps) {

    const router = useRouter();

    // If the user email is set, we add it to the form as a hidden input
    if (userEmail) {
        return (
            <div className="flex justify-center">
                <form action="/api/checkout_sessions" method="POST">
                    {/* <input type="hidden" name="user_id" value={userData.user.id} /> */}
                    {userEmail && <input type="hidden" name="user_email" value={userEmail} />}
                    <button
                        onClick={onClick}
                        className={`bg-primary-green hover:bg-dark-gray text-dark-gray hover:text-primary-green font-semibold py-3 px-12 rounded-lg transition-colors duration-200 ${className}`}
                        type="submit"
                    >
                        {children}
                    </button>
                </form>
            </div>
        );
    }
    // Otherwise, when clicked, the user is sent to the Auth page to sign in or sign up
    else {
        return (
            <div className="flex justify-center">
                <button
                    className={`bg-primary-green hover:bg-dark-gray text-dark-gray hover:text-primary-green font-semibold py-3 px-12 rounded-lg transition-colors duration-200 ${className}`}
                    onClick={() => {
                        router.push('/auth/login');
                    }}>
                    {children}
                </button>
            </div>
        );
    }
}
