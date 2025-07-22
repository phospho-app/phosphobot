"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProtectedRoute = ProtectedRoute;
const jsx_runtime_1 = require("react/jsx-runtime");
const AuthContext_1 = require("../context/AuthContext");
function ProtectedRoute({ children, fallback = (0, jsx_runtime_1.jsx)("div", { children: "Loading..." }), requireProUser = false }) {
    const { session, isLoading, proUser } = (0, AuthContext_1.useAuth)();
    if (isLoading) {
        return (0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: fallback });
    }
    if (!session) {
        return (0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: fallback });
    }
    if (requireProUser && !proUser) {
        return (0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: fallback });
    }
    return (0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: children });
}
