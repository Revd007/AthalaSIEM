"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("app/page",{

/***/ "(app-pages-browser)/./src/app/page.tsx":
/*!**************************!*\
  !*** ./src/app/page.tsx ***!
  \**************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ Login)\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"(app-pages-browser)/./node_modules/next/dist/compiled/react/jsx-dev-runtime.js\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ \"(app-pages-browser)/./node_modules/next/dist/compiled/react/index.js\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var next_navigation__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/navigation */ \"(app-pages-browser)/./node_modules/next/dist/api/navigation.js\");\n/* __next_internal_client_entry_do_not_use__ default auto */ \nvar _s = $RefreshSig$();\n\n\n\nfunction Login() {\n    _s();\n    const router = (0,next_navigation__WEBPACK_IMPORTED_MODULE_2__.useRouter)();\n    const [error, setError] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)('');\n    const [formData, setFormData] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)({\n        username: '',\n        password: ''\n    });\n    const handleSubmit = async (e)=>{\n        e.preventDefault();\n        setError('');\n        try {\n            var _data_user;\n            const response = await fetch('/api/auth//routeslogin', {\n                method: 'POST',\n                headers: {\n                    'Content-Type': 'application/json'\n                },\n                body: JSON.stringify({\n                    username: formData.username,\n                    password: formData.password\n                })\n            });\n            if (!response.ok) {\n                const errorData = await response.json().catch(()=>({}));\n                throw new Error(errorData.message || 'Login failed. Please try again.');\n            }\n            const data = await response.json();\n            localStorage.setItem('token', data.token);\n            if ((_data_user = data.user) === null || _data_user === void 0 ? void 0 : _data_user.role) {\n                localStorage.setItem('userRole', data.user.role);\n            }\n            router.push('/dashboard');\n        } catch (error) {\n            console.error('Login error:', error);\n            setError(error instanceof Error ? error.message : 'An unexpected error occurred');\n        }\n    };\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n        className: \"min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8\",\n        children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n            className: \"max-w-md w-full space-y-8\",\n            children: [\n                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                    children: [\n                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"img\", {\n                            className: \"mx-auto h-12 w-auto\",\n                            src: \"/assets/images/logo.svg\",\n                            alt: \"SIEM Logo\"\n                        }, void 0, false, {\n                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                            lineNumber: 55,\n                            columnNumber: 11\n                        }, this),\n                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"h2\", {\n                            className: \"mt-6 text-center text-3xl font-extrabold text-gray-900\",\n                            children: \"Sign in to your account\"\n                        }, void 0, false, {\n                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                            lineNumber: 60,\n                            columnNumber: 11\n                        }, this)\n                    ]\n                }, void 0, true, {\n                    fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                    lineNumber: 54,\n                    columnNumber: 9\n                }, this),\n                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"form\", {\n                    className: \"mt-8 space-y-6\",\n                    onSubmit: handleSubmit,\n                    children: [\n                        error && /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                            className: \"rounded-md bg-red-50 p-4\",\n                            children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                                className: \"text-sm text-red-700\",\n                                children: error\n                            }, void 0, false, {\n                                fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                lineNumber: 67,\n                                columnNumber: 15\n                            }, this)\n                        }, void 0, false, {\n                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                            lineNumber: 66,\n                            columnNumber: 13\n                        }, this),\n                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                            className: \"rounded-md shadow-sm -space-y-px\",\n                            children: [\n                                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                                    children: [\n                                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"label\", {\n                                            htmlFor: \"username\",\n                                            className: \"sr-only\",\n                                            children: \"Username\"\n                                        }, void 0, false, {\n                                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                            lineNumber: 72,\n                                            columnNumber: 15\n                                        }, this),\n                                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"input\", {\n                                            id: \"username\",\n                                            name: \"username\",\n                                            type: \"text\",\n                                            required: true,\n                                            className: \"appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm\",\n                                            placeholder: \"Username\",\n                                            value: formData.username,\n                                            onChange: (e)=>setFormData({\n                                                    ...formData,\n                                                    username: e.target.value\n                                                })\n                                        }, void 0, false, {\n                                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                            lineNumber: 75,\n                                            columnNumber: 15\n                                        }, this)\n                                    ]\n                                }, void 0, true, {\n                                    fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                    lineNumber: 71,\n                                    columnNumber: 13\n                                }, this),\n                                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                                    children: [\n                                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"label\", {\n                                            htmlFor: \"password\",\n                                            className: \"sr-only\",\n                                            children: \"Password\"\n                                        }, void 0, false, {\n                                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                            lineNumber: 87,\n                                            columnNumber: 15\n                                        }, this),\n                                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"input\", {\n                                            id: \"password\",\n                                            name: \"password\",\n                                            type: \"password\",\n                                            required: true,\n                                            className: \"appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm\",\n                                            placeholder: \"Password\",\n                                            value: formData.password,\n                                            onChange: (e)=>setFormData({\n                                                    ...formData,\n                                                    password: e.target.value\n                                                })\n                                        }, void 0, false, {\n                                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                            lineNumber: 90,\n                                            columnNumber: 15\n                                        }, this)\n                                    ]\n                                }, void 0, true, {\n                                    fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                    lineNumber: 86,\n                                    columnNumber: 13\n                                }, this)\n                            ]\n                        }, void 0, true, {\n                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                            lineNumber: 70,\n                            columnNumber: 11\n                        }, this),\n                        /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                            children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"button\", {\n                                type: \"submit\",\n                                className: \"group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500\",\n                                children: \"Sign in\"\n                            }, void 0, false, {\n                                fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                                lineNumber: 104,\n                                columnNumber: 13\n                            }, this)\n                        }, void 0, false, {\n                            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                            lineNumber: 103,\n                            columnNumber: 11\n                        }, this)\n                    ]\n                }, void 0, true, {\n                    fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n                    lineNumber: 64,\n                    columnNumber: 9\n                }, this)\n            ]\n        }, void 0, true, {\n            fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n            lineNumber: 53,\n            columnNumber: 7\n        }, this)\n    }, void 0, false, {\n        fileName: \"E:\\\\AthalaSIEM\\\\AthalaSIEM\\\\frontend\\\\siem-frontend\\\\src\\\\app\\\\page.tsx\",\n        lineNumber: 52,\n        columnNumber: 5\n    }, this);\n}\n_s(Login, \"3UZKqFakV8O3dEmpFyORAgMpFH4=\", false, function() {\n    return [\n        next_navigation__WEBPACK_IMPORTED_MODULE_2__.useRouter\n    ];\n});\n_c = Login;\nvar _c;\n$RefreshReg$(_c, \"Login\");\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,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QUFNLENBQUNGLEtBQUs7Z0RBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7OztzQ0FLM0UsOERBQUNoQjtzQ0FDQyw0RUFBQ21CO2dDQUNDTixNQUFLO2dDQUNMWixXQUFVOzBDQUNYOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBUWI7R0EzR3dCakM7O1FBQ1BGLHNEQUFTQTs7O0tBREZFIiwic291cmNlcyI6WyJFOlxcQXRoYWxhU0lFTVxcQXRoYWxhU0lFTVxcZnJvbnRlbmRcXHNpZW0tZnJvbnRlbmRcXHNyY1xcYXBwXFxwYWdlLnRzeCJdLCJzb3VyY2VzQ29udGVudCI6WyIndXNlIGNsaWVudCdcclxuXHJcbmltcG9ydCB7IHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnXHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvbmF2aWdhdGlvbidcclxuaW1wb3J0IExpbmsgZnJvbSAnbmV4dC9saW5rJ1xyXG5pbXBvcnQgUmVhY3QgZnJvbSAncmVhY3QnXHJcblxyXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBMb2dpbigpIHtcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKVxyXG4gIGNvbnN0IFtlcnJvciwgc2V0RXJyb3JdID0gdXNlU3RhdGUoJycpXHJcbiAgY29uc3QgW2Zvcm1EYXRhLCBzZXRGb3JtRGF0YV0gPSB1c2VTdGF0ZSh7XHJcbiAgICB1c2VybmFtZTogJycsXHJcbiAgICBwYXNzd29yZDogJycsXHJcbiAgfSlcclxuXHJcbiAgY29uc3QgaGFuZGxlU3VibWl0ID0gYXN5bmMgKGU6IFJlYWN0LkZvcm1FdmVudCkgPT4ge1xyXG4gICAgZS5wcmV2ZW50RGVmYXVsdCgpXHJcbiAgICBzZXRFcnJvcignJylcclxuICAgIFxyXG4gICAgdHJ5IHtcclxuICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBmZXRjaCgnL2FwaS9hdXRoLy9yb3V0ZXNsb2dpbicsIHtcclxuICAgICAgICBtZXRob2Q6ICdQT1NUJyxcclxuICAgICAgICBoZWFkZXJzOiB7XHJcbiAgICAgICAgICAnQ29udGVudC1UeXBlJzogJ2FwcGxpY2F0aW9uL2pzb24nLFxyXG4gICAgICAgIH0sXHJcbiAgICAgICAgYm9keTogSlNPTi5zdHJpbmdpZnkoe1xyXG4gICAgICAgICAgdXNlcm5hbWU6IGZvcm1EYXRhLnVzZXJuYW1lLFxyXG4gICAgICAgICAgcGFzc3dvcmQ6IGZvcm1EYXRhLnBhc3N3b3JkLFxyXG4gICAgICAgIH0pLFxyXG4gICAgICB9KVxyXG5cclxuICAgICAgaWYgKCFyZXNwb25zZS5vaykge1xyXG4gICAgICAgIGNvbnN0IGVycm9yRGF0YSA9IGF3YWl0IHJlc3BvbnNlLmpzb24oKS5jYXRjaCgoKSA9PiAoe30pKVxyXG4gICAgICAgIHRocm93IG5ldyBFcnJvcihlcnJvckRhdGEubWVzc2FnZSB8fCAnTG9naW4gZmFpbGVkLiBQbGVhc2UgdHJ5IGFnYWluLicpXHJcbiAgICAgIH1cclxuXHJcbiAgICAgIGNvbnN0IGRhdGEgPSBhd2FpdCByZXNwb25zZS5qc29uKClcclxuICAgICAgbG9jYWxTdG9yYWdlLnNldEl0ZW0oJ3Rva2VuJywgZGF0YS50b2tlbilcclxuICAgICAgXHJcbiAgICAgIGlmIChkYXRhLnVzZXI/LnJvbGUpIHtcclxuICAgICAgICBsb2NhbFN0b3JhZ2Uuc2V0SXRlbSgndXNlclJvbGUnLCBkYXRhLnVzZXIucm9sZSlcclxuICAgICAgfVxyXG4gICAgICBcclxuICAgICAgcm91dGVyLnB1c2goJy9kYXNoYm9hcmQnKVxyXG4gICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgY29uc29sZS5lcnJvcignTG9naW4gZXJyb3I6JywgZXJyb3IpXHJcbiAgICAgIHNldEVycm9yKGVycm9yIGluc3RhbmNlb2YgRXJyb3IgPyBlcnJvci5tZXNzYWdlIDogJ0FuIHVuZXhwZWN0ZWQgZXJyb3Igb2NjdXJyZWQnKVxyXG4gICAgfVxyXG4gIH1cclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxkaXYgY2xhc3NOYW1lPVwibWluLWgtc2NyZWVuIGZsZXggaXRlbXMtY2VudGVyIGp1c3RpZnktY2VudGVyIGJnLWdyYXktNTAgcHktMTIgcHgtNCBzbTpweC02IGxnOnB4LThcIj5cclxuICAgICAgPGRpdiBjbGFzc05hbWU9XCJtYXgtdy1tZCB3LWZ1bGwgc3BhY2UteS04XCI+XHJcbiAgICAgICAgPGRpdj5cclxuICAgICAgICAgIDxpbWdcclxuICAgICAgICAgICAgY2xhc3NOYW1lPVwibXgtYXV0byBoLTEyIHctYXV0b1wiXHJcbiAgICAgICAgICAgIHNyYz1cIi9hc3NldHMvaW1hZ2VzL2xvZ28uc3ZnXCJcclxuICAgICAgICAgICAgYWx0PVwiU0lFTSBMb2dvXCJcclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8aDIgY2xhc3NOYW1lPVwibXQtNiB0ZXh0LWNlbnRlciB0ZXh0LTN4bCBmb250LWV4dHJhYm9sZCB0ZXh0LWdyYXktOTAwXCI+XHJcbiAgICAgICAgICAgIFNpZ24gaW4gdG8geW91ciBhY2NvdW50XHJcbiAgICAgICAgICA8L2gyPlxyXG4gICAgICAgIDwvZGl2PlxyXG4gICAgICAgIDxmb3JtIGNsYXNzTmFtZT1cIm10LTggc3BhY2UteS02XCIgb25TdWJtaXQ9e2hhbmRsZVN1Ym1pdH0+XHJcbiAgICAgICAgICB7ZXJyb3IgJiYgKFxyXG4gICAgICAgICAgICA8ZGl2IGNsYXNzTmFtZT1cInJvdW5kZWQtbWQgYmctcmVkLTUwIHAtNFwiPlxyXG4gICAgICAgICAgICAgIDxkaXYgY2xhc3NOYW1lPVwidGV4dC1zbSB0ZXh0LXJlZC03MDBcIj57ZXJyb3J9PC9kaXY+XHJcbiAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICAgIDxkaXYgY2xhc3NOYW1lPVwicm91bmRlZC1tZCBzaGFkb3ctc20gLXNwYWNlLXktcHhcIj5cclxuICAgICAgICAgICAgPGRpdj5cclxuICAgICAgICAgICAgICA8bGFiZWwgaHRtbEZvcj1cInVzZXJuYW1lXCIgY2xhc3NOYW1lPVwic3Itb25seVwiPlxyXG4gICAgICAgICAgICAgICAgVXNlcm5hbWVcclxuICAgICAgICAgICAgICA8L2xhYmVsPlxyXG4gICAgICAgICAgICAgIDxpbnB1dFxyXG4gICAgICAgICAgICAgICAgaWQ9XCJ1c2VybmFtZVwiXHJcbiAgICAgICAgICAgICAgICBuYW1lPVwidXNlcm5hbWVcIlxyXG4gICAgICAgICAgICAgICAgdHlwZT1cInRleHRcIlxyXG4gICAgICAgICAgICAgICAgcmVxdWlyZWRcclxuICAgICAgICAgICAgICAgIGNsYXNzTmFtZT1cImFwcGVhcmFuY2Utbm9uZSByb3VuZGVkLW5vbmUgcmVsYXRpdmUgYmxvY2sgdy1mdWxsIHB4LTMgcHktMiBib3JkZXIgYm9yZGVyLWdyYXktMzAwIHBsYWNlaG9sZGVyLWdyYXktNTAwIHRleHQtZ3JheS05MDAgcm91bmRlZC10LW1kIGZvY3VzOm91dGxpbmUtbm9uZSBmb2N1czpyaW5nLWluZGlnby01MDAgZm9jdXM6Ym9yZGVyLWluZGlnby01MDAgZm9jdXM6ei0xMCBzbTp0ZXh0LXNtXCJcclxuICAgICAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiVXNlcm5hbWVcIlxyXG4gICAgICAgICAgICAgICAgdmFsdWU9e2Zvcm1EYXRhLnVzZXJuYW1lfVxyXG4gICAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlKSA9PiBzZXRGb3JtRGF0YSh7IC4uLmZvcm1EYXRhLCB1c2VybmFtZTogZS50YXJnZXQudmFsdWUgfSl9XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICAgIDxkaXY+XHJcbiAgICAgICAgICAgICAgPGxhYmVsIGh0bWxGb3I9XCJwYXNzd29yZFwiIGNsYXNzTmFtZT1cInNyLW9ubHlcIj5cclxuICAgICAgICAgICAgICAgIFBhc3N3b3JkXHJcbiAgICAgICAgICAgICAgPC9sYWJlbD5cclxuICAgICAgICAgICAgICA8aW5wdXRcclxuICAgICAgICAgICAgICAgIGlkPVwicGFzc3dvcmRcIlxyXG4gICAgICAgICAgICAgICAgbmFtZT1cInBhc3N3b3JkXCJcclxuICAgICAgICAgICAgICAgIHR5cGU9XCJwYXNzd29yZFwiXHJcbiAgICAgICAgICAgICAgICByZXF1aXJlZFxyXG4gICAgICAgICAgICAgICAgY2xhc3NOYW1lPVwiYXBwZWFyYW5jZS1ub25lIHJvdW5kZWQtbm9uZSByZWxhdGl2ZSBibG9jayB3LWZ1bGwgcHgtMyBweS0yIGJvcmRlciBib3JkZXItZ3JheS0zMDAgcGxhY2Vob2xkZXItZ3JheS01MDAgdGV4dC1ncmF5LTkwMCByb3VuZGVkLWItbWQgZm9jdXM6b3V0bGluZS1ub25lIGZvY3VzOnJpbmctaW5kaWdvLTUwMCBmb2N1czpib3JkZXItaW5kaWdvLTUwMCBmb2N1czp6LTEwIHNtOnRleHQtc21cIlxyXG4gICAgICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJQYXNzd29yZFwiXHJcbiAgICAgICAgICAgICAgICB2YWx1ZT17Zm9ybURhdGEucGFzc3dvcmR9XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGUpID0+IHNldEZvcm1EYXRhKHsgLi4uZm9ybURhdGEsIHBhc3N3b3JkOiBlLnRhcmdldC52YWx1ZSB9KX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgIDwvZGl2PlxyXG5cclxuICAgICAgICAgIDxkaXY+XHJcbiAgICAgICAgICAgIDxidXR0b25cclxuICAgICAgICAgICAgICB0eXBlPVwic3VibWl0XCJcclxuICAgICAgICAgICAgICBjbGFzc05hbWU9XCJncm91cCByZWxhdGl2ZSB3LWZ1bGwgZmxleCBqdXN0aWZ5LWNlbnRlciBweS0yIHB4LTQgYm9yZGVyIGJvcmRlci10cmFuc3BhcmVudCB0ZXh0LXNtIGZvbnQtbWVkaXVtIHJvdW5kZWQtbWQgdGV4dC13aGl0ZSBiZy1pbmRpZ28tNjAwIGhvdmVyOmJnLWluZGlnby03MDAgZm9jdXM6b3V0bGluZS1ub25lIGZvY3VzOnJpbmctMiBmb2N1czpyaW5nLW9mZnNldC0yIGZvY3VzOnJpbmctaW5kaWdvLTUwMFwiXHJcbiAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICBTaWduIGluXHJcbiAgICAgICAgICAgIDwvYnV0dG9uPlxyXG4gICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgPC9mb3JtPlxyXG4gICAgICA8L2Rpdj5cclxuICAgIDwvZGl2PlxyXG4gIClcclxufSJdLCJuYW1lcyI6WyJ1c2VTdGF0ZSIsInVzZVJvdXRlciIsIlJlYWN0IiwiTG9naW4iLCJyb3V0ZXIiLCJlcnJvciIsInNldEVycm9yIiwiZm9ybURhdGEiLCJzZXRGb3JtRGF0YSIsInVzZXJuYW1lIiwicGFzc3dvcmQiLCJoYW5kbGVTdWJtaXQiLCJlIiwicHJldmVudERlZmF1bHQiLCJkYXRhIiwicmVzcG9uc2UiLCJmZXRjaCIsIm1ldGhvZCIsImhlYWRlcnMiLCJib2R5IiwiSlNPTiIsInN0cmluZ2lmeSIsIm9rIiwiZXJyb3JEYXRhIiwianNvbiIsImNhdGNoIiwiRXJyb3IiLCJtZXNzYWdlIiwibG9jYWxTdG9yYWdlIiwic2V0SXRlbSIsInRva2VuIiwidXNlciIsInJvbGUiLCJwdXNoIiwiY29uc29sZSIsImRpdiIsImNsYXNzTmFtZSIsImltZyIsInNyYyIsImFsdCIsImgyIiwiZm9ybSIsIm9uU3VibWl0IiwibGFiZWwiLCJodG1sRm9yIiwiaW5wdXQiLCJpZCIsIm5hbWUiLCJ0eXBlIiwicmVxdWlyZWQiLCJwbGFjZWhvbGRlciIsInZhbHVlIiwib25DaGFuZ2UiLCJ0YXJnZXQiLCJidXR0b24iXSwiaWdub3JlTGlzdCI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(app-pages-browser)/./src/app/page.tsx\n"));

/***/ })

});