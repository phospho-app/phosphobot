"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AuthForm = AuthForm;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const AuthContext_1 = require("../context/AuthContext");
function AuthForm({ mode = 'signin', onSuccess, className = '' }) {
    const [email, setEmail] = (0, react_1.useState)('');
    const [password, setPassword] = (0, react_1.useState)('');
    const [isLoading, setIsLoading] = (0, react_1.useState)(false);
    const { login, signup } = (0, AuthContext_1.useAuth)();
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        if (!email || !password) {
            setIsLoading(false);
            return;
        }
        try {
            if (mode === 'signup') {
                await signup(email, password);
            }
            else {
                await login(email, password);
            }
            onSuccess?.();
        }
        catch (error) {
            // Error is handled in context
        }
        finally {
            setIsLoading(false);
        }
    };
    return ((0, jsx_runtime_1.jsxs)("form", { onSubmit: handleSubmit, className: `flex flex-col gap-4 ${className}`, children: [(0, jsx_runtime_1.jsx)("input", { type: "email", value: email, onChange: (e) => setEmail(e.target.value), placeholder: "Email", className: "border p-2 rounded", required: true, disabled: isLoading }), (0, jsx_runtime_1.jsx)("input", { type: "password", value: password, onChange: (e) => setPassword(e.target.value), placeholder: "Password", className: "border p-2 rounded", required: true, disabled: isLoading }), (0, jsx_runtime_1.jsx)("button", { type: "submit", className: "bg-blue-500 text-white p-2 rounded disabled:opacity-50", disabled: isLoading, children: isLoading ? 'Processing...' : (mode === 'signup' ? 'Sign Up' : 'Sign In') })] }));
}
