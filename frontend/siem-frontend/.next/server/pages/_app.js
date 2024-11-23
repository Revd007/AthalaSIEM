/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/_app";
exports.ids = ["pages/_app"];
exports.modules = {

/***/ "./src/hooks/use-auth.ts":
/*!*******************************!*\
  !*** ./src/hooks/use-auth.ts ***!
  \*******************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   useAuth: () => (/* binding */ useAuth)\n/* harmony export */ });\n/* harmony import */ var zustand__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! zustand */ \"zustand\");\n/* harmony import */ var _lib_axios__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../lib/axios */ \"./src/lib/axios.ts\");\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([zustand__WEBPACK_IMPORTED_MODULE_0__, _lib_axios__WEBPACK_IMPORTED_MODULE_1__]);\n([zustand__WEBPACK_IMPORTED_MODULE_0__, _lib_axios__WEBPACK_IMPORTED_MODULE_1__] = __webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__);\n// src/hooks/use-auth.ts\n\n\nconst useAuth = (0,zustand__WEBPACK_IMPORTED_MODULE_0__.create)((set)=>({\n        user: null,\n        token: localStorage.getItem('token'),\n        isLoading: true,\n        login: async (credentials)=>{\n            try {\n                const response = await _lib_axios__WEBPACK_IMPORTED_MODULE_1__.axiosInstance.post('/auth/login', credentials);\n                const { token, user } = response.data;\n                localStorage.setItem('token', token);\n                set({\n                    user,\n                    token\n                });\n            } catch (error) {\n                console.error('Login failed:', error);\n                throw error;\n            }\n        },\n        logout: ()=>{\n            localStorage.removeItem('token');\n            set({\n                user: null,\n                token: null\n            });\n        },\n        checkAuth: async ()=>{\n            try {\n                const token = localStorage.getItem('token');\n                if (!token) {\n                    set({\n                        isLoading: false\n                    });\n                    return;\n                }\n                const response = await _lib_axios__WEBPACK_IMPORTED_MODULE_1__.axiosInstance.get('/auth/me');\n                set({\n                    user: response.data,\n                    isLoading: false\n                });\n            } catch (error) {\n                localStorage.removeItem('token');\n                set({\n                    user: null,\n                    token: null,\n                    isLoading: false\n                });\n            }\n        }\n    }));\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9zcmMvaG9va3MvdXNlLWF1dGgudHMiLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7O0FBQUEsd0JBQXdCO0FBQ1E7QUFDWTtBQVdyQyxNQUFNRSxVQUFVRiwrQ0FBTUEsQ0FBWSxDQUFDRyxNQUFTO1FBQ2pEQyxNQUFNO1FBQ05DLE9BQU9DLGFBQWFDLE9BQU8sQ0FBQztRQUM1QkMsV0FBVztRQUVYQyxPQUFPLE9BQU9DO1lBQ1osSUFBSTtnQkFDRixNQUFNQyxXQUFXLE1BQU1WLHFEQUFhQSxDQUFDVyxJQUFJLENBQUMsZUFBZUY7Z0JBQ3pELE1BQU0sRUFBRUwsS0FBSyxFQUFFRCxJQUFJLEVBQUUsR0FBR08sU0FBU0UsSUFBSTtnQkFDckNQLGFBQWFRLE9BQU8sQ0FBQyxTQUFTVDtnQkFDOUJGLElBQUk7b0JBQUVDO29CQUFNQztnQkFBTTtZQUNwQixFQUFFLE9BQU9VLE9BQU87Z0JBQ2RDLFFBQVFELEtBQUssQ0FBQyxpQkFBaUJBO2dCQUMvQixNQUFNQTtZQUNSO1FBQ0Y7UUFFQUUsUUFBUTtZQUNOWCxhQUFhWSxVQUFVLENBQUM7WUFDeEJmLElBQUk7Z0JBQUVDLE1BQU07Z0JBQU1DLE9BQU87WUFBSztRQUNoQztRQUVBYyxXQUFXO1lBQ1QsSUFBSTtnQkFDRixNQUFNZCxRQUFRQyxhQUFhQyxPQUFPLENBQUM7Z0JBQ25DLElBQUksQ0FBQ0YsT0FBTztvQkFDVkYsSUFBSTt3QkFBRUssV0FBVztvQkFBTTtvQkFDdkI7Z0JBQ0Y7Z0JBRUEsTUFBTUcsV0FBVyxNQUFNVixxREFBYUEsQ0FBQ21CLEdBQUcsQ0FBQztnQkFDekNqQixJQUFJO29CQUFFQyxNQUFNTyxTQUFTRSxJQUFJO29CQUFFTCxXQUFXO2dCQUFNO1lBQzlDLEVBQUUsT0FBT08sT0FBTztnQkFDZFQsYUFBYVksVUFBVSxDQUFDO2dCQUN4QmYsSUFBSTtvQkFBRUMsTUFBTTtvQkFBTUMsT0FBTztvQkFBTUcsV0FBVztnQkFBTTtZQUNsRDtRQUNGO0lBQ0YsSUFBRyIsInNvdXJjZXMiOlsiRTpcXEF0aGFsYVNJRU1cXEF0aGFsYVNJRU1cXGZyb250ZW5kXFxzaWVtLWZyb250ZW5kXFxzcmNcXGhvb2tzXFx1c2UtYXV0aC50cyJdLCJzb3VyY2VzQ29udGVudCI6WyIvLyBzcmMvaG9va3MvdXNlLWF1dGgudHNcclxuaW1wb3J0IHsgY3JlYXRlIH0gZnJvbSAnenVzdGFuZCdcclxuaW1wb3J0IHsgYXhpb3NJbnN0YW5jZSB9IGZyb20gJy4uL2xpYi9heGlvcydcclxuXHJcbmludGVyZmFjZSBBdXRoU3RhdGUge1xyXG4gIHVzZXI6IGFueSB8IG51bGxcclxuICB0b2tlbjogc3RyaW5nIHwgbnVsbFxyXG4gIGlzTG9hZGluZzogYm9vbGVhblxyXG4gIGxvZ2luOiAoY3JlZGVudGlhbHM6IHsgdXNlcm5hbWU6IHN0cmluZzsgcGFzc3dvcmQ6IHN0cmluZyB9KSA9PiBQcm9taXNlPHZvaWQ+XHJcbiAgbG9nb3V0OiAoKSA9PiB2b2lkXHJcbiAgY2hlY2tBdXRoOiAoKSA9PiBQcm9taXNlPHZvaWQ+XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCB1c2VBdXRoID0gY3JlYXRlPEF1dGhTdGF0ZT4oKHNldCkgPT4gKHtcclxuICB1c2VyOiBudWxsLFxyXG4gIHRva2VuOiBsb2NhbFN0b3JhZ2UuZ2V0SXRlbSgndG9rZW4nKSxcclxuICBpc0xvYWRpbmc6IHRydWUsXHJcbiAgXHJcbiAgbG9naW46IGFzeW5jIChjcmVkZW50aWFscykgPT4ge1xyXG4gICAgdHJ5IHtcclxuICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBheGlvc0luc3RhbmNlLnBvc3QoJy9hdXRoL2xvZ2luJywgY3JlZGVudGlhbHMpXHJcbiAgICAgIGNvbnN0IHsgdG9rZW4sIHVzZXIgfSA9IHJlc3BvbnNlLmRhdGFcclxuICAgICAgbG9jYWxTdG9yYWdlLnNldEl0ZW0oJ3Rva2VuJywgdG9rZW4pXHJcbiAgICAgIHNldCh7IHVzZXIsIHRva2VuIH0pXHJcbiAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICBjb25zb2xlLmVycm9yKCdMb2dpbiBmYWlsZWQ6JywgZXJyb3IpXHJcbiAgICAgIHRocm93IGVycm9yXHJcbiAgICB9XHJcbiAgfSxcclxuICBcclxuICBsb2dvdXQ6ICgpID0+IHtcclxuICAgIGxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKCd0b2tlbicpXHJcbiAgICBzZXQoeyB1c2VyOiBudWxsLCB0b2tlbjogbnVsbCB9KVxyXG4gIH0sXHJcbiAgXHJcbiAgY2hlY2tBdXRoOiBhc3luYyAoKSA9PiB7XHJcbiAgICB0cnkge1xyXG4gICAgICBjb25zdCB0b2tlbiA9IGxvY2FsU3RvcmFnZS5nZXRJdGVtKCd0b2tlbicpXHJcbiAgICAgIGlmICghdG9rZW4pIHtcclxuICAgICAgICBzZXQoeyBpc0xvYWRpbmc6IGZhbHNlIH0pXHJcbiAgICAgICAgcmV0dXJuXHJcbiAgICAgIH1cclxuICAgICAgXHJcbiAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgYXhpb3NJbnN0YW5jZS5nZXQoJy9hdXRoL21lJylcclxuICAgICAgc2V0KHsgdXNlcjogcmVzcG9uc2UuZGF0YSwgaXNMb2FkaW5nOiBmYWxzZSB9KVxyXG4gICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgbG9jYWxTdG9yYWdlLnJlbW92ZUl0ZW0oJ3Rva2VuJylcclxuICAgICAgc2V0KHsgdXNlcjogbnVsbCwgdG9rZW46IG51bGwsIGlzTG9hZGluZzogZmFsc2UgfSlcclxuICAgIH1cclxuICB9LFxyXG59KSkiXSwibmFtZXMiOlsiY3JlYXRlIiwiYXhpb3NJbnN0YW5jZSIsInVzZUF1dGgiLCJzZXQiLCJ1c2VyIiwidG9rZW4iLCJsb2NhbFN0b3JhZ2UiLCJnZXRJdGVtIiwiaXNMb2FkaW5nIiwibG9naW4iLCJjcmVkZW50aWFscyIsInJlc3BvbnNlIiwicG9zdCIsImRhdGEiLCJzZXRJdGVtIiwiZXJyb3IiLCJjb25zb2xlIiwibG9nb3V0IiwicmVtb3ZlSXRlbSIsImNoZWNrQXV0aCIsImdldCJdLCJpZ25vcmVMaXN0IjpbXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///./src/hooks/use-auth.ts\n");

/***/ }),

/***/ "./src/lib/axios.ts":
/*!**************************!*\
  !*** ./src/lib/axios.ts ***!
  \**************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   axiosInstance: () => (/* binding */ axiosInstance)\n/* harmony export */ });\n/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! axios */ \"axios\");\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([axios__WEBPACK_IMPORTED_MODULE_0__]);\naxios__WEBPACK_IMPORTED_MODULE_0__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];\n\nconst axiosInstance = axios__WEBPACK_IMPORTED_MODULE_0__[\"default\"].create({\n    baseURL: \"http://localhost:8000/api\" || 0,\n    timeout: 10000,\n    headers: {\n        'Content-Type': 'application/json'\n    }\n});\n// Add request interceptor\naxiosInstance.interceptors.request.use((config)=>{\n    const token = localStorage.getItem('token');\n    if (token) {\n        config.headers.Authorization = `Bearer ${token}`;\n    }\n    return config;\n}, (error)=>{\n    return Promise.reject(error);\n});\n// Add response interceptor\naxiosInstance.interceptors.response.use((response)=>response, async (error)=>{\n    if (error.response?.status === 401) {\n        // Handle token expiration\n        localStorage.removeItem('token');\n        window.location.href = '/login';\n    }\n    return Promise.reject(error);\n});\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9zcmMvbGliL2F4aW9zLnRzIiwibWFwcGluZ3MiOiI7Ozs7Ozs7O0FBQTBCO0FBRW5CLE1BQU1DLGdCQUFnQkQsb0RBQVksQ0FBQztJQUN4Q0csU0FBU0MsMkJBQStCLElBQUksQ0FBMkI7SUFDdkVHLFNBQVM7SUFDVEMsU0FBUztRQUNQLGdCQUFnQjtJQUNsQjtBQUNGLEdBQUc7QUFFSCwwQkFBMEI7QUFDMUJQLGNBQWNRLFlBQVksQ0FBQ0MsT0FBTyxDQUFDQyxHQUFHLENBQ3BDLENBQUNDO0lBQ0MsTUFBTUMsUUFBUUMsYUFBYUMsT0FBTyxDQUFDO0lBQ25DLElBQUlGLE9BQU87UUFDVEQsT0FBT0osT0FBTyxDQUFDUSxhQUFhLEdBQUcsQ0FBQyxPQUFPLEVBQUVILE9BQU87SUFDbEQ7SUFDQSxPQUFPRDtBQUNULEdBQ0EsQ0FBQ0s7SUFDQyxPQUFPQyxRQUFRQyxNQUFNLENBQUNGO0FBQ3hCO0FBR0YsMkJBQTJCO0FBQzNCaEIsY0FBY1EsWUFBWSxDQUFDVyxRQUFRLENBQUNULEdBQUcsQ0FDckMsQ0FBQ1MsV0FBYUEsVUFDZCxPQUFPSDtJQUNMLElBQUlBLE1BQU1HLFFBQVEsRUFBRUMsV0FBVyxLQUFLO1FBQ2xDLDBCQUEwQjtRQUMxQlAsYUFBYVEsVUFBVSxDQUFDO1FBQ3hCQyxPQUFPQyxRQUFRLENBQUNDLElBQUksR0FBRztJQUN6QjtJQUNBLE9BQU9QLFFBQVFDLE1BQU0sQ0FBQ0Y7QUFDeEIiLCJzb3VyY2VzIjpbIkU6XFxBdGhhbGFTSUVNXFxBdGhhbGFTSUVNXFxmcm9udGVuZFxcc2llbS1mcm9udGVuZFxcc3JjXFxsaWJcXGF4aW9zLnRzIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBheGlvcyBmcm9tICdheGlvcyc7XHJcblxyXG5leHBvcnQgY29uc3QgYXhpb3NJbnN0YW5jZSA9IGF4aW9zLmNyZWF0ZSh7XHJcbiAgYmFzZVVSTDogcHJvY2Vzcy5lbnYuTkVYVF9QVUJMSUNfQVBJX1VSTCB8fCAnaHR0cDovL2xvY2FsaG9zdDo4MDAwL2FwaScsXHJcbiAgdGltZW91dDogMTAwMDAsXHJcbiAgaGVhZGVyczoge1xyXG4gICAgJ0NvbnRlbnQtVHlwZSc6ICdhcHBsaWNhdGlvbi9qc29uJyxcclxuICB9LFxyXG59KTtcclxuXHJcbi8vIEFkZCByZXF1ZXN0IGludGVyY2VwdG9yXHJcbmF4aW9zSW5zdGFuY2UuaW50ZXJjZXB0b3JzLnJlcXVlc3QudXNlKFxyXG4gIChjb25maWcpID0+IHtcclxuICAgIGNvbnN0IHRva2VuID0gbG9jYWxTdG9yYWdlLmdldEl0ZW0oJ3Rva2VuJyk7XHJcbiAgICBpZiAodG9rZW4pIHtcclxuICAgICAgY29uZmlnLmhlYWRlcnMuQXV0aG9yaXphdGlvbiA9IGBCZWFyZXIgJHt0b2tlbn1gO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIGNvbmZpZztcclxuICB9LFxyXG4gIChlcnJvcikgPT4ge1xyXG4gICAgcmV0dXJuIFByb21pc2UucmVqZWN0KGVycm9yKTtcclxuICB9XHJcbik7XHJcblxyXG4vLyBBZGQgcmVzcG9uc2UgaW50ZXJjZXB0b3JcclxuYXhpb3NJbnN0YW5jZS5pbnRlcmNlcHRvcnMucmVzcG9uc2UudXNlKFxyXG4gIChyZXNwb25zZSkgPT4gcmVzcG9uc2UsXHJcbiAgYXN5bmMgKGVycm9yKSA9PiB7XHJcbiAgICBpZiAoZXJyb3IucmVzcG9uc2U/LnN0YXR1cyA9PT0gNDAxKSB7XHJcbiAgICAgIC8vIEhhbmRsZSB0b2tlbiBleHBpcmF0aW9uXHJcbiAgICAgIGxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKCd0b2tlbicpO1xyXG4gICAgICB3aW5kb3cubG9jYXRpb24uaHJlZiA9ICcvbG9naW4nO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIFByb21pc2UucmVqZWN0KGVycm9yKTtcclxuICB9XHJcbik7Il0sIm5hbWVzIjpbImF4aW9zIiwiYXhpb3NJbnN0YW5jZSIsImNyZWF0ZSIsImJhc2VVUkwiLCJwcm9jZXNzIiwiZW52IiwiTkVYVF9QVUJMSUNfQVBJX1VSTCIsInRpbWVvdXQiLCJoZWFkZXJzIiwiaW50ZXJjZXB0b3JzIiwicmVxdWVzdCIsInVzZSIsImNvbmZpZyIsInRva2VuIiwibG9jYWxTdG9yYWdlIiwiZ2V0SXRlbSIsIkF1dGhvcml6YXRpb24iLCJlcnJvciIsIlByb21pc2UiLCJyZWplY3QiLCJyZXNwb25zZSIsInN0YXR1cyIsInJlbW92ZUl0ZW0iLCJ3aW5kb3ciLCJsb2NhdGlvbiIsImhyZWYiXSwiaWdub3JlTGlzdCI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./src/lib/axios.ts\n");

/***/ }),

/***/ "./src/pages/_app.tsx":
/*!****************************!*\
  !*** ./src/pages/_app.tsx ***!
  \****************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ App)\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"react/jsx-dev-runtime\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var _hooks_use_auth__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../hooks/use-auth */ \"./src/hooks/use-auth.ts\");\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../styles/globals.css */ \"./src/styles/globals.css\");\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_styles_globals_css__WEBPACK_IMPORTED_MODULE_2__);\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([_hooks_use_auth__WEBPACK_IMPORTED_MODULE_1__]);\n_hooks_use_auth__WEBPACK_IMPORTED_MODULE_1__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];\n\n\n\nfunction App({ Component, pageProps }) {\n    const { isLoading } = (0,_hooks_use_auth__WEBPACK_IMPORTED_MODULE_1__.useAuth)();\n    if (isLoading) {\n        return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n            children: \"Loading...\"\n        }, void 0, false, {\n            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\pages\\\\_app.tsx\",\n            lineNumber: 9,\n            columnNumber: 12\n        }, this);\n    }\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(Component, {\n        ...pageProps\n    }, void 0, false, {\n        fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\pages\\\\_app.tsx\",\n        lineNumber: 12,\n        columnNumber: 10\n    }, this);\n}\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9zcmMvcGFnZXMvX2FwcC50c3giLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7OztBQUMyQztBQUNiO0FBRWYsU0FBU0MsSUFBSSxFQUFFQyxTQUFTLEVBQUVDLFNBQVMsRUFBWTtJQUM1RCxNQUFNLEVBQUVDLFNBQVMsRUFBRSxHQUFHSix3REFBT0E7SUFFN0IsSUFBSUksV0FBVztRQUNiLHFCQUFPLDhEQUFDQztzQkFBSTs7Ozs7O0lBQ2Q7SUFFQSxxQkFBTyw4REFBQ0g7UUFBVyxHQUFHQyxTQUFTOzs7Ozs7QUFDakMiLCJzb3VyY2VzIjpbIkU6XFxBdGhhbGFTSUVNXFxBdGhhbGFTSUVNXFxmcm9udGVuZFxcc2llbS1mcm9udGVuZFxcc3JjXFxwYWdlc1xcX2FwcC50c3giXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHR5cGUgeyBBcHBQcm9wcyB9IGZyb20gJ25leHQvYXBwJ1xyXG5pbXBvcnQgeyB1c2VBdXRoIH0gZnJvbSAnLi4vaG9va3MvdXNlLWF1dGgnXHJcbmltcG9ydCAnLi4vc3R5bGVzL2dsb2JhbHMuY3NzJ1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gQXBwKHsgQ29tcG9uZW50LCBwYWdlUHJvcHMgfTogQXBwUHJvcHMpIHtcclxuICBjb25zdCB7IGlzTG9hZGluZyB9ID0gdXNlQXV0aCgpXHJcblxyXG4gIGlmIChpc0xvYWRpbmcpIHtcclxuICAgIHJldHVybiA8ZGl2PkxvYWRpbmcuLi48L2Rpdj5cclxuICB9XHJcblxyXG4gIHJldHVybiA8Q29tcG9uZW50IHsuLi5wYWdlUHJvcHN9IC8+XHJcbn1cclxuIl0sIm5hbWVzIjpbInVzZUF1dGgiLCJBcHAiLCJDb21wb25lbnQiLCJwYWdlUHJvcHMiLCJpc0xvYWRpbmciLCJkaXYiXSwiaWdub3JlTGlzdCI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./src/pages/_app.tsx\n");

/***/ }),

/***/ "./src/styles/globals.css":
/*!********************************!*\
  !*** ./src/styles/globals.css ***!
  \********************************/
/***/ (() => {



/***/ }),

/***/ "react/jsx-dev-runtime":
/*!****************************************!*\
  !*** external "react/jsx-dev-runtime" ***!
  \****************************************/
/***/ ((module) => {

"use strict";
module.exports = require("react/jsx-dev-runtime");

/***/ }),

/***/ "axios":
/*!************************!*\
  !*** external "axios" ***!
  \************************/
/***/ ((module) => {

"use strict";
module.exports = import("axios");;

/***/ }),

/***/ "zustand":
/*!**************************!*\
  !*** external "zustand" ***!
  \**************************/
/***/ ((module) => {

"use strict";
module.exports = import("zustand");;

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = (__webpack_exec__("./src/pages/_app.tsx"));
module.exports = __webpack_exports__;

})();