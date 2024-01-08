import { ThemeProvider, createTheme } from '@mui/material/styles';
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { CssBaseline } from '@mui/material';

const darkTheme = createTheme({
    palette: {
        mode: 'dark',
    },
})

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(
    <React.StrictMode>
        <ThemeProvider theme={darkTheme}>
           <CssBaseline /> 
           <App />
        </ThemeProvider>
    </React.StrictMode>

)
