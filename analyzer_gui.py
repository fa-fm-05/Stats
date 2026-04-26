"""
Advanced Data Analyzer GUI
--------------------------
A professional-grade data analysis application built with CustomTkinter, Pandas, and Matplotlib.
Implements a clean architecture separating the user interface (View), data processing (Model),
and visualization (Plotter) components. Modernized with segmented controls and custom frames.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import logging
import textwrap
from typing import Optional, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a nicer plotting style
matplotlib.use('TkAgg')

# CustomTkinter setup
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class Theme:
    """Application theme constants for Matplotlib."""
    # Plot colors (Tailored for dark mode)
    plot_bg: str = '#2b2b2b'
    plot_face: str = '#1e1e1e'
    plot_text: str = '#ffffff'
    plot_grid: str = '#444444'
    
    plot_bar: str = '#1f6aa5'
    plot_bar_edge: str = '#144870'
    plot_line_trend: str = '#e74c3c'
    plot_line_freq: str = '#2ecc71'
    plot_hist: str = '#9b59b6'
    plot_scatter: str = '#f39c12'
    plot_scatter_edge: str = '#000000'
    plot_trendline: str = '#c0392b'


class DataModel:
    """Handles all data loading and statistical operations."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filename: str = ""

    def load_data(self, filepath: str) -> bool:
        """Loads data from a CSV or Excel file."""
        try:
            if filepath.lower().endswith('.csv'):
                self.df = pd.read_csv(filepath)
            else:
                self.df = pd.read_excel(filepath)
            self.filename = filepath.split('/')[-1]
            logging.info(f"Successfully loaded {self.filename} with {len(self.df)} rows.")
            return True
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise Exception(f"Error loading file: {str(e)}")

    def get_columns(self) -> List[str]:
        """Returns a list of column names."""
        return list(self.df.columns) if self.df is not None else []

    def get_column_data(self, column: str) -> pd.Series:
        """Returns non-null data for a specific column."""
        if self.df is None or column not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[column].dropna()

    def get_bivariate_data(self, col1: str, col2: str) -> pd.DataFrame:
        """Returns non-null data for two columns."""
        if self.df is None or col1 not in self.df.columns or col2 not in self.df.columns:
            return pd.DataFrame()
        return self.df[[col1, col2]].dropna()

    def is_numeric(self, column: str) -> bool:
        """Checks if a column is numeric."""
        if self.df is None or column not in self.df.columns:
            return False
        return pd.api.types.is_numeric_dtype(self.df[column])

    def calculate_univariate_stats(self, column: str) -> str:
        """Calculates and formats statistical summary for a column."""
        data = self.get_column_data(column)
        if data.empty:
            return "No valid data to analyze."

        stats_str = f"=== Analysis for '{column}' ===\n\n"
        
        if self.is_numeric(column):
            stats_str += "📊 [Numerical Stats]\n"
            stats_str += f"Count   : {len(data)}\n"
            stats_str += f"Mean    : {data.mean():.4f}\n"
            stats_str += f"Median  : {data.median():.4f}\n"
            stats_str += f"Std Dev : {data.std():.4f}\n"
            stats_str += f"Variance: {data.var():.4f}\n\n"
        else:
            stats_str += "⚠️ (Non-numeric column, numerical stats skipped)\n\n"
            
        freq = data.value_counts()
        perc = data.value_counts(normalize=True) * 100
        
        stats_str += "📋 [Frequency & Percentage Distribution]\n"
        for val in freq.index[:10]:
            stats_str += f" • {val}: {freq[val]} ({perc[val]:.1f}%)\n"
            
        if len(freq) > 10:
            stats_str += f"... (and {len(freq) - 10} more values)"
            
        return stats_str


class Plotter:
    """Encapsulates Matplotlib plotting logic."""
    
    def __init__(self, theme: Theme):
        self.theme = theme

    def _setup_figure(self) -> Tuple[Figure, matplotlib.axes.Axes]:
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        fig.patch.set_facecolor(self.theme.plot_bg)
        ax.set_facecolor(self.theme.plot_face)
        
        ax.tick_params(colors=self.theme.plot_text)
        for spine in ax.spines.values():
            spine.set_color(self.theme.plot_text)
            
        ax.xaxis.label.set_color(self.theme.plot_text)
        ax.yaxis.label.set_color(self.theme.plot_text)
        ax.title.set_color(self.theme.plot_text)
        
        ax.grid(True, linestyle='--', alpha=0.3, color=self.theme.plot_grid)
        return fig, ax

    def _wrap_title(self, title: str) -> str:
        """Wraps long titles so they don't squash the graph."""
        return textwrap.fill(title, width=50)

    def create_bar_chart(self, data: pd.Series, title: str) -> Figure:
        fig, ax = self._setup_figure()
        freq_data = data.value_counts().head(15)
        
        freq_data.plot(kind='bar', ax=ax, color=self.theme.plot_bar, 
                      edgecolor=self.theme.plot_bar_edge, alpha=0.9)
        ax.set_title(self._wrap_title(f"Bar Chart: {title}"))
        ax.set_ylabel("Frequency")
        # Wrap long labels
        labels = [textwrap.fill(str(label), width=15) for label in freq_data.index]
        ax.set_xticklabels(labels)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        return fig

    def create_pie_chart(self, data: pd.Series, title: str) -> Figure:
        fig, ax = self._setup_figure()
        ax.grid(False) # Turn off grid for pie chart
        freq_data = data.value_counts().head(10)
        
        # Wrap long labels
        labels = [textwrap.fill(str(label), width=15) for label in freq_data.index]
        
        freq_data.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, 
                      labels=labels,
                      wedgeprops={'edgecolor': 'black'}, 
                      textprops={'color': 'white'})
        ax.set_title(self._wrap_title(f"Pie Chart: {title}"))
        ax.set_ylabel("")
        fig.tight_layout()
        return fig

    def create_line_chart(self, data: pd.Series, title: str, is_numeric: bool) -> Figure:
        fig, ax = self._setup_figure()
        
        if is_numeric:
            ax.plot(data.values, color=self.theme.plot_line_trend, linewidth=2, 
                   marker='o', markersize=3, alpha=0.8)
            ax.set_title(self._wrap_title(f"Line Chart (Trend): {title}"))
            ax.set_ylabel("Value")
            ax.set_xlabel("Index")
        else:
            freq = data.value_counts().sort_index()
            ax.plot(freq.index.astype(str), freq.values, color=self.theme.plot_line_freq, 
                   linewidth=2, marker='s', markersize=5)
            ax.set_title(self._wrap_title(f"Line Chart (Frequency): {title}"))
            ax.set_ylabel("Frequency")
            fig.autofmt_xdate(rotation=45)
            
        fig.tight_layout()
        return fig

    def create_histogram(self, data: pd.Series, title: str) -> Figure:
        fig, ax = self._setup_figure()
        
        bins = min(20, len(data.unique()))
        ax.hist(data, bins=bins, color=self.theme.plot_hist, edgecolor='white', alpha=0.85)
        
        ax.set_title(self._wrap_title(f"Histogram: {title}"))
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Value Range")
        fig.tight_layout()
        return fig

    def create_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str) -> Figure:
        fig, ax = self._setup_figure()
        
        corr = df[col1].corr(df[col2])
        
        ax.scatter(df[col1], df[col2], alpha=0.7, color=self.theme.plot_scatter, 
                  edgecolors=self.theme.plot_scatter_edge, s=50)
        
        if len(df) > 1:
            z = np.polyfit(df[col1], df[col2], 1)
            p = np.poly1d(z)
            ax.plot(df[col1], p(df[col1]), color=self.theme.plot_trendline, 
                   linestyle="--", linewidth=2.5)
            
        ax.set_title(self._wrap_title(f"Scatter Plot (Pearson r = {corr:.3f})"))
        ax.set_xlabel(textwrap.fill(col1, width=30))
        ax.set_ylabel(textwrap.fill(col2, width=30))
        fig.tight_layout()
        return fig


class DataAnalyzerApp(ctk.CTk):
    """Main Application built with CustomTkinter."""
    
    def __init__(self):
        super().__init__()
        
        self.theme = Theme()
        self.model = DataModel()
        self.plotter = Plotter(self.theme)
        
        self.current_canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        
        # State variables for popping out graphs
        self.last_plot_type = None
        self.last_plot_kwargs = {}
        
        self.title("Advanced Data Analyzer Pro")
        self.geometry("1200x850")
        self.minsize(900, 600)
        
        self._build_layout()
        
    def _build_layout(self):
        # Configure 1x2 grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1) # Push things up
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Data Analyzer Pro", 
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # 1. Data Source Frame
        self.data_frame = ctk.CTkFrame(self.sidebar_frame)
        self.data_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.lbl_data = ctk.CTkLabel(self.data_frame, text="1. Data Source", font=ctk.CTkFont(weight="bold"))
        self.lbl_data.pack(anchor="w", padx=10, pady=(10, 0))
        
        self.btn_load = ctk.CTkButton(self.data_frame, text="📁 Load CSV / Excel", command=self._handle_load_data)
        self.btn_load.pack(fill="x", padx=10, pady=10)
        
        self.lbl_file = ctk.CTkLabel(self.data_frame, text="No dataset loaded", text_color="gray")
        self.lbl_file.pack(anchor="w", padx=10, pady=(0, 10))

        # Analysis Mode Segmented Button
        self.mode_var = ctk.StringVar(value="Univariate")
        self.seg_button = ctk.CTkSegmentedButton(self.sidebar_frame, 
                                                 values=["Univariate", "Bivariate"],
                                                 variable=self.mode_var,
                                                 command=self._switch_analysis_mode)
        self.seg_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # 2. Univariate Frame
        self.uni_frame = ctk.CTkFrame(self.sidebar_frame)
        self.uni_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.uni_frame, text="Target Column:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.cb_column = ctk.CTkOptionMenu(self.uni_frame, values=["(Select Data First)"], command=self._handle_column_select)
        self.cb_column.pack(fill="x", padx=10, pady=10)
        
        # Grid for chart buttons
        self.btn_grid = ctk.CTkFrame(self.uni_frame, fg_color="transparent")
        self.btn_grid.pack(fill="x", padx=5, pady=5)
        self.btn_grid.grid_columnconfigure((0, 1), weight=1)
        
        self.btn_bar = ctk.CTkButton(self.btn_grid, text="📊 Bar", command=lambda: self._handle_plot('bar'))
        self.btn_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_pie = ctk.CTkButton(self.btn_grid, text="🥧 Pie", command=lambda: self._handle_plot('pie'))
        self.btn_pie.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_line = ctk.CTkButton(self.btn_grid, text="📈 Line", command=lambda: self._handle_plot('line'))
        self.btn_line.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_hist = ctk.CTkButton(self.btn_grid, text="📉 Hist", command=lambda: self._handle_plot('hist'))
        self.btn_hist.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # 3. Bivariate Frame (Initially Hidden)
        self.bi_frame = ctk.CTkFrame(self.sidebar_frame)
        
        ctk.CTkLabel(self.bi_frame, text="X-Axis Variable:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.cb_col1 = ctk.CTkOptionMenu(self.bi_frame, values=["(Select Data First)"])
        self.cb_col1.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.bi_frame, text="Y-Axis Variable:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(0, 0))
        self.cb_col2 = ctk.CTkOptionMenu(self.bi_frame, values=["(Select Data First)"])
        self.cb_col2.pack(fill="x", padx=10, pady=10)
        
        self.btn_scatter = ctk.CTkButton(self.bi_frame, text="🔵 Generate Scatter Plot", command=self._handle_scatter_plot)
        self.btn_scatter.pack(fill="x", padx=10, pady=(10, 15))

        # Appearance Settings
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("Dark")

        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Stats Output
        self.stats_frame = ctk.CTkFrame(self.main_frame)
        self.stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        ctk.CTkLabel(self.stats_frame, text="Statistical Summary", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        self.txt_stats = ctk.CTkTextbox(self.stats_frame, height=180, font=ctk.CTkFont(family="Consolas", size=13))
        self.txt_stats.pack(fill="x", padx=15, pady=(0, 15))
        self._update_stats_text("Please load a dataset to begin analysis.")
        
        # Graph Area
        self.graph_container = ctk.CTkFrame(self.main_frame)
        self.graph_container.grid(row=1, column=0, sticky="nsew")
        self.graph_container.grid_rowconfigure(1, weight=1)
        self.graph_container.grid_columnconfigure(0, weight=1)
        
        # Graph Header with Pop Out Button
        self.graph_header = ctk.CTkFrame(self.graph_container, fg_color="transparent")
        self.graph_header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.graph_title_lbl = ctk.CTkLabel(self.graph_header, text="Visualizations Dashboard", font=ctk.CTkFont(weight="bold"))
        self.graph_title_lbl.pack(side="left")
        
        self.btn_popout = ctk.CTkButton(self.graph_header, text="⛶ Pop Out Graph", width=120, 
                                        command=self._popout_graph, state="disabled")
        self.btn_popout.pack(side="right")
        
        # Inner frame for the actual canvas
        self.graph_frame = ctk.CTkFrame(self.graph_container, fg_color="transparent")
        self.graph_frame.grid(row=1, column=0, sticky="nsew")
        
        self.empty_graph_lbl = ctk.CTkLabel(self.graph_frame, text="Graphs will appear here.", 
                                         font=ctk.CTkFont(size=20, weight="bold"), text_color="gray")
        self.empty_graph_lbl.pack(expand=True)

    # --- UI Logic ---
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        # Note: True dynamic matplotlib dark/light mode switching requires replotting
        # For simplicity, we just change the UI elements here.

    def _switch_analysis_mode(self, mode: str):
        if mode == "Univariate":
            self.bi_frame.grid_forget()
            self.uni_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        else:
            self.uni_frame.grid_forget()
            self.bi_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

    def _update_stats_text(self, text: str):
        self.txt_stats.configure(state="normal")
        self.txt_stats.delete("1.0", "end")
        self.txt_stats.insert("end", text)
        self.txt_stats.configure(state="disabled")

    def _handle_load_data(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Data Files", "*.csv *.xlsx *.xls"), ("All Files", "*.*")]
        )
        if not filepath:
            return
            
        try:
            self.config(cursor="watch")
            self.update()
            
            if self.model.load_data(filepath):
                self.lbl_file.configure(text=f"Loaded: {self.model.filename}", text_color="green")
                self._populate_comboboxes()
                self._update_stats_text(f"Successfully loaded '{self.model.filename}'\nDataset contains {len(self.model.df)} rows and {len(self.model.df.columns)} columns.")
                self._clear_graph()
                
        except Exception as e:
            messagebox.showerror("Data Loading Error", str(e))
        finally:
            self.config(cursor="")

    def _populate_comboboxes(self):
        columns = self.model.get_columns()
        if columns:
            self.cb_column.configure(values=columns)
            self.cb_column.set(columns[0])
            self.cb_col1.configure(values=columns)
            self.cb_col1.set(columns[0])
            self.cb_col2.configure(values=columns)
            self.cb_col2.set(columns[-1] if len(columns) > 1 else columns[0])
            self._handle_column_select(columns[0])

    def _handle_column_select(self, col: str):
        if not col or col == "(Select Data First)":
            return
            
        stats_str = self.model.calculate_univariate_stats(col)
        self._update_stats_text(stats_str)

    def _clear_graph(self):
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
            
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
            
        self.empty_graph_lbl.pack(expand=True)
        self.btn_popout.configure(state="disabled")

    def _render_figure(self, fig: Figure):
        """Renders a matplotlib figure onto the Tkinter canvas with navigation tools."""
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
            
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
            
        self.empty_graph_lbl.pack_forget()
        self.btn_popout.configure(state="normal")
        
        # We don't call tight_layout here since Plotter methods do it, 
        # and it can conflict if called multiple times.
        
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        
        widget = self.current_canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        # Add Navigation Toolbar (Scaling/Panning Gadgets)
        self.toolbar = NavigationToolbar2Tk(self.current_canvas, self.graph_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

    def _generate_plot(self, plot_type: str, kwargs: dict) -> Optional[Figure]:
        """Helper to generate a plot based on saved state, used for main view and popouts."""
        try:
            if plot_type == 'bar':
                return self.plotter.create_bar_chart(kwargs['data'], kwargs['col'])
            elif plot_type == 'pie':
                return self.plotter.create_pie_chart(kwargs['data'], kwargs['col'])
            elif plot_type == 'line':
                return self.plotter.create_line_chart(kwargs['data'], kwargs['col'], kwargs['is_num'])
            elif plot_type == 'hist':
                return self.plotter.create_histogram(kwargs['data'], kwargs['col'])
            elif plot_type == 'scatter':
                return self.plotter.create_scatter_plot(kwargs['df_bi'], kwargs['col1'], kwargs['col2'])
        except Exception as e:
            logging.error(f"Plotting error: {e}", exc_info=True)
            messagebox.showerror("Plotting Error", f"An error occurred while generating the plot:\n{e}")
        return None

    def _handle_plot(self, plot_type: str):
        col = self.cb_column.get()
        if not col or col == "(Select Data First)" or self.model.df is None:
            messagebox.showinfo("Information", "Please select a target column first.")
            return
            
        data = self.model.get_column_data(col)
        if data.empty:
            messagebox.showwarning("Warning", f"No valid data found in column '{col}'.")
            return

        if plot_type == 'hist' and not self.model.is_numeric(col):
            messagebox.showerror("Error", "Histogram requires a numeric column.")
            return

        # Save state for popout functionality
        self.last_plot_type = plot_type
        self.last_plot_kwargs = {'data': data, 'col': col, 'is_num': self.model.is_numeric(col)}

        fig = self._generate_plot(self.last_plot_type, self.last_plot_kwargs)
        if fig:
            self._render_figure(fig)

    def _handle_scatter_plot(self):
        col1 = self.cb_col1.get()
        col2 = self.cb_col2.get()
        
        if not col1 or not col2 or col1 == "(Select Data First)" or self.model.df is None:
            messagebox.showinfo("Information", "Please select both X and Y axis variables.")
            return
            
        if not self.model.is_numeric(col1) or not self.model.is_numeric(col2):
            messagebox.showerror("Error", "Both variables must be numeric to generate a scatter plot.")
            return
            
        df_bi = self.model.get_bivariate_data(col1, col2)
        if df_bi.empty:
            messagebox.showwarning("Warning", "No overlapping valid data found for the selected columns.")
            return

        # Save state for popout functionality
        self.last_plot_type = 'scatter'
        self.last_plot_kwargs = {'df_bi': df_bi, 'col1': col1, 'col2': col2}

        fig = self._generate_plot(self.last_plot_type, self.last_plot_kwargs)
        if fig:
            self._render_figure(fig)

    def _popout_graph(self):
        """Creates a new top-level window with a copy of the current graph."""
        if not self.last_plot_type:
            return
            
        popout_win = ctk.CTkToplevel(self)
        popout_win.title("Expanded Graph View")
        popout_win.geometry("800x600")
        
        # Redraw the figure for the new canvas
        fig = self._generate_plot(self.last_plot_type, self.last_plot_kwargs)
        if not fig:
            return
            
        canvas = FigureCanvasTkAgg(fig, master=popout_win)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        # Add toolbar to popout as well
        toolbar = NavigationToolbar2Tk(canvas, popout_win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x", padx=10, pady=(0, 10))


if __name__ == "__main__":
    app = DataAnalyzerApp()
    app.mainloop()
