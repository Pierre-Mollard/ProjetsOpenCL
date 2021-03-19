// Mandelbrot.cpp : Définit le point d'entrée de l'application.
//

#include "framework.h"
#include "Mandelbrot.h"
#include <CL/opencl.h>

#define MAX_LOADSTRING 100


const char* KernelSource = "\n" \
"__kernel void mandel(                                                    \n" \
"   const double x0,                                                    \n" \
"   const double y0,                                                    \n" \
"   const double stepsize,                                                    \n" \
"   const unsigned int maxIter,                                         \n" \
"   __global unsigned int *restrict framebuffer,                        \n" \
"   const unsigned int windowWidth                                      \n" \
"   )                                                                   \n" \
"{// WORK ITEM POSITION                                                                      \n" \
"   const size_t windowPosX = get_global_id(0);                                          \n" \
"   const size_t windowPosY = get_global_id(1);                                            \n" \
"   const double stepPosX = x0 + (windowPosX * stepsize);                                           \n" \
"   const double stepPosY = y0 - (windowPosY * stepsize);                                           \n" \
"                                              \n" \
"   double x = 0.0;                                          \n" \
"   double y = 0.0;                                      \n" \
"   double x2 = 0.0;                                        \n" \
"   double y2 = 0.0;                                        \n" \
"   unsigned int i = 0;                                           \n" \
"                                              \n" \
"   while(x2 + y2 < 4.0 && i < maxIter){                                           \n" \
"        x2 = x*x;                                      \n" \
"        y2 = y*y;                                      \n" \
"        y = 2*x*y + stepPosY;                                      \n" \
"        x = x2 - y2 + stepPosX;                                      \n" \
"        i++;                          }            \n" \
"                    \n" \
"   framebuffer[windowWidth * windowPosY + windowPosX] = i%16;                                           \n" \
"}                                                                      \n" \
"\n";

// Variables globales :
HINSTANCE hInst;                                // instance actuelle
WCHAR szTitle[MAX_LOADSTRING];                  // Texte de la barre de titre
WCHAR szWindowClass[MAX_LOADSTRING];            // nom de la classe de fenêtre principale

int specialTest = 1;

// Déclarations anticipées des fonctions incluses dans ce module de code :
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Placez le code ici.
    //
    //
    //


    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_platform_id platform_id;
    cl_uint num_of_platform = 0;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem input, output;
    size_t global;

    //retrieves a list of platforms available
    if (clGetPlatformIDs(1, &platform_id, &num_of_platform) != CL_SUCCESS)
    {
        return 1;
    }
    //try to get supported GPU devices
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
    {
        return 1;
    }
    //context properties list - must be terminated with 0
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform_id;
    properties[2] = 0;
    // Initialise les chaînes globales
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_MANDELBROT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);
    //create a context wuth the GPU Device
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    //create a command queue using the context and device
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    //create a program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    //compile the program
    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    {
        return 1;
    }
    //specify which kernel from the program to execute
    kernel = clCreateKernel(program, "mandel", &err);


    specialTest++;

    // Effectue l'initialisation de l'application :
    if (!InitInstance (hInstance, nCmdShow))
    {

        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_MANDELBROT));

    MSG msg;

    // Boucle de messages principale :
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FONCTION : MyRegisterClass()
//
//  OBJECTIF : Inscrit la classe de fenêtre.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MANDELBROT));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_MANDELBROT);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FONCTION : InitInstance(HINSTANCE, int)
//
//   OBJECTIF : enregistre le handle d'instance et crée une fenêtre principale
//
//   COMMENTAIRES :
//
//        Dans cette fonction, nous enregistrons le handle de l'instance dans une variable globale, puis
//        nous créons et affichons la fenêtre principale du programme.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Stocke le handle d'instance dans la variable globale

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FONCTION : WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  OBJECTIF : Traite les messages pour la fenêtre principale.
//
//  WM_COMMAND  - traite le menu de l'application
//  WM_PAINT    - Dessine la fenêtre principale
//  WM_DESTROY  - génère un message d'arrêt et retourne
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Analyse les sélections de menu :
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Ajoutez ici le code de dessin qui utilise hdc...

            
            
            SetPixel(hdc, specialTest, 10, RGB(122, 122, 122));


            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Gestionnaire de messages pour la boîte de dialogue À propos de.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
