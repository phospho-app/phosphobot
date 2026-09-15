"use client";

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
  userEmail,
}: SubscribeButtonProps) {

  // If the user email is set, show the checkout form (the API reads the user from the session)
  if (userEmail) {
    return (
      <div className="flex justify-center">
        <form action="/api/checkout_sessions" method="POST">
          <button
            onClick={onClick}
            className={`bg-gray-400 text-gray-600 font-semibold py-3 px-12 rounded-lg cursor-not-allowed ${className}`}
            type="submit"
            disabled
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
          className={`bg-gray-400 text-gray-600 font-semibold py-3 px-12 rounded-lg cursor-not-allowed ${className}`}
          disabled
        >
          {children}
        </button>
      </div>
    );
  }
}
