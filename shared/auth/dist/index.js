"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProtectedRoute = exports.AuthForm = exports.useAuth = exports.AuthProvider = exports.SupabaseAuthProvider = exports.DashboardAuthProvider = void 0;
// Types
__exportStar(require("./types"), exports);
// Providers
var dashboard_1 = require("./providers/dashboard");
Object.defineProperty(exports, "DashboardAuthProvider", { enumerable: true, get: function () { return dashboard_1.DashboardAuthProvider; } });
var supabase_1 = require("./providers/supabase");
Object.defineProperty(exports, "SupabaseAuthProvider", { enumerable: true, get: function () { return supabase_1.SupabaseAuthProvider; } });
// Context
var AuthContext_1 = require("./context/AuthContext");
Object.defineProperty(exports, "AuthProvider", { enumerable: true, get: function () { return AuthContext_1.AuthProvider; } });
Object.defineProperty(exports, "useAuth", { enumerable: true, get: function () { return AuthContext_1.useAuth; } });
// Components
var AuthForm_1 = require("./components/AuthForm");
Object.defineProperty(exports, "AuthForm", { enumerable: true, get: function () { return AuthForm_1.AuthForm; } });
var ProtectedRoute_1 = require("./components/ProtectedRoute");
Object.defineProperty(exports, "ProtectedRoute", { enumerable: true, get: function () { return ProtectedRoute_1.ProtectedRoute; } });
