// Mandelbrot.cpp : Définit le point d'entrée de l'application.
//

#include "framework.h"
#include "Mandelbrot.h"
#include <CL/opencl.h>
#include <iostream>
#include <time.h>

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
"    int mod = i%16;                                                \n" \
"    char r, g, b = 0;                                                \n" \
"    switch (mod)                                                \n" \
"    {                                                                             \n" \
"        case 0: r = 66; g = 30; b = 15; break;                                                \n" \
"        case 1: r = 25; g = 7; b = 26; break;                                                \n" \
"        case 2: r = 9; g = 1; b = 47; break;                                                \n" \
"        case 3: r = 4; g = 4; b = 73; break;                                                \n" \
"        case 4: r = 0; g = 7; b = 100; break;                                                \n" \
"        case 5: r = 12; g = 44; b = 138; break;                                                \n" \
"        case 6: r = 24; g = 82; b = 177; break;                                                \n" \
"        case 7: r = 57; g = 125; b = 209; break;                                                \n" \
"        case 8: r = 134; g = 181; b = 229; break;                                                \n" \
"        case 9: r = 211; g = 236; b = 248; break;                                                \n" \
"        case 10: r = 241; g = 233; b = 191; break;                                                \n" \
"        case 11: r = 248; g = 201; b = 95; break;                                                \n" \
"        case 12: r = 254; g = 170; b = 0; break;                                                \n" \
"        case 13: r = 204; g = 128; b = 0; break;                                                \n" \
"        case 14: r = 153; g = 87; b = 0; break;                                                \n" \
"        case 15: r = 106; g = 52; b = 3; break;                                                \n" \
"    }                                                                                                       \n" \
"   framebuffer[windowWidth * windowPosY + windowPosX] = (unsigned int)(0 + (r<<16) + (g<<8) + b);                 \n" \
"}                                                                      \n" \
"\n";

// Variables globales :
HINSTANCE hInst;                                // instance actuelle
WCHAR szTitle[MAX_LOADSTRING];                  // Texte de la barre de titre
WCHAR szWindowClass[MAX_LOADSTRING];            // nom de la classe de fenêtre principale
HWND hTextInput;           // nom de la classe de fenêtre principale
HWND hTextOutput;
HDC hdc;
void* memory;
unsigned int* grid;
int imgWIDTH = 1000;
int imgHEIGHT = 1000;
int gridOffsetX = 300;
int gridOffsetY = 10;

// Déclarations anticipées des fonctions incluses dans ce module de code :
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

BITMAPINFO bitmap_info;


void saveBMP(const char* name, int width, int height, unsigned int* data, int maxIter) {
    FILE* f;

    unsigned int headers[13];
    int extrabytes = 4 - ((width * 3) % 4);
    if (extrabytes == 4)
        extrabytes = 0;
    int paddedsize = ((width * 3) + extrabytes) * height;

    headers[0] = paddedsize + 54;
    headers[1] = 0;
    headers[2] = 54;
    headers[3] = 40;
    headers[4] = width;
    headers[5] = height;

    headers[7] = 0;
    headers[8] = paddedsize;
    headers[9] = 0;
    headers[10] = 0;
    headers[11] = 0;
    headers[12] = 0;

    fopen_s(&f, name, "wb");

    int n;
    fprintf(f, "BM");
    for (n = 0; n <= 5; n++) {
        fprintf(f, "%c", headers[n] & 0x000000FF);
        fprintf(f, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(f, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(f, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    fprintf(f, "%c", 1);
    fprintf(f, "%c", 0);
    fprintf(f, "%c", 24);
    fprintf(f, "%c", 0);

    for (n = 7; n <= 12; n++) {
        fprintf(f, "%c", headers[n] & 0x000000FF);
        fprintf(f, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(f, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(f, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    int iter = -1;
    int x, y;

    for (y = height - 1; y >= 0; y--) {
        for (x = 0; x < width; x++) {
            iter++;
            int mod = (unsigned char)*(data + iter);
            int r, g, b = 0;
            switch (mod)
            {
            case 0: r = 66; g = 30; b = 15; break;
            case 1: r = 25; g = 7; b = 26; break;
            case 2: r = 9; g = 1; b = 47; break;
            case 3: r = 4; g = 4; b = 73; break;
            case 4: r = 0; g = 7; b = 100; break;
            case 5: r = 12; g = 44; b = 138; break;
            case 6: r = 24; g = 82; b = 177; break;
            case 7: r = 57; g = 125; b = 209; break;
            case 8: r = 134; g = 181; b = 229; break;
            case 9: r = 211; g = 236; b = 248; break;
            case 10: r = 241; g = 233; b = 191; break;
            case 11: r = 248; g = 201; b = 95; break;
            case 12: r = 254; g = 170; b = 0; break;
            case 13: r = 204; g = 128; b = 0; break;
            case 14: r = 153; g = 87; b = 0; break;
            case 15: r = 106; g = 52; b = 3; break;
            }

            if (r < 0 || r> 255)
                r = 0;

            if (g < 0 || g> 255)
                g = 0;

            if (b < 0 || b> 255)
                b = 0;

            fprintf(f, "%c", b);
            fprintf(f, "%c", g);
            fprintf(f, "%c", r);
        }
        if (extrabytes) {
            for (n = 1; n <= extrabytes; n++) {
                fprintf(f, "%c", 0);
            }
        }
    }
    fclose(f);
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    bitmap_info.bmiHeader.biSize = sizeof(bitmap_info.bmiHeader);
    bitmap_info.bmiHeader.biWidth = imgWIDTH;
    bitmap_info.bmiHeader.biHeight = imgHEIGHT;
    bitmap_info.bmiHeader.biPlanes = 1;
    bitmap_info.bmiHeader.biBitCount = 32;
    bitmap_info.bmiHeader.biCompression = BI_RGB;

    
    // TODO: Placez le code ici.

    size_t global;                      // global domain size  
    size_t local;                       // local  domain size  

    double step;
    int nworkgroup = 32;
    int max_size = 16;
    int workgroup_size = max_size;
    int i;

    int ntotal_iter = imgHEIGHT * imgWIDTH;

    float xmax = 1.5;
    float xmin = -2;
    float ymax = 1.75f;
    float ymin = -1.75f;

    double startX = -2;
    double startY = 1.75;

    int maxIter = 255;

    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue commands;
    cl_program program;
    cl_int err;
    cl_platform_id platform_id;
    cl_uint num_of_platform = 0;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem y_out;    

    grid = (unsigned int*)malloc(imgWIDTH * imgHEIGHT * sizeof(unsigned int));
    //memory = VirtualAlloc(grid, imgWIDTH * imgHEIGHT * sizeof(unsigned int), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

    clGetPlatformIDs(1, &platform_id, &num_of_platform);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);

    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform_id;
    properties[2] = 0;


    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "mandel", &err);

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_size), &max_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    }
    if (max_size > workgroup_size) workgroup_size = max_size;

    // Now that we know the size of the work_groups, we can set the number of work
    // groups, the actual number of steps, and the step size
    nworkgroup = ntotal_iter / (workgroup_size);

    if (nworkgroup < 1)
    {
        int comp_units;
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to access device number of compute units !\n");
            return EXIT_FAILURE;
        }
        nworkgroup = comp_units;
        workgroup_size = ntotal_iter / (nworkgroup);
    }
    int nsteps = workgroup_size * nworkgroup;
    step = 1.0 / (double)nsteps;
    step = 0.0025;

    printf("Total iter : %d || Nstep : %d\n", ntotal_iter, nsteps);
    printf(" %d work groups of size %d.  %d Integration steps\n", (int)nworkgroup, (int)workgroup_size, nsteps);

    y_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * imgHEIGHT * imgWIDTH, NULL, NULL);

    // Set the arguments to our compute kernel
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_double), &startX);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_double), &startY);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_double), &step);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &maxIter);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_out);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &imgWIDTH);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    }
    double rtime;
    rtime = clock();

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    cl_event prof_event;
    global = nworkgroup * workgroup_size;
    local = workgroup_size;
    size_t ggg[2] = { imgWIDTH, imgHEIGHT };
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, ggg, NULL, 0, NULL, &prof_event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the commands to complete before reading back results
    clFinish(commands);


    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, y_out, CL_TRUE, 0, sizeof(unsigned int) * imgHEIGHT * imgWIDTH, grid, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // extract timing data from the event, prof_event
    err = clWaitForEvents(1, &prof_event);
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    wchar_t * tempL[32];
    //swprintf_s(&tempL[32], L"prof says %f ms \n", (double)(ev_end_time - ev_start_time) * 1.0e-6);
    

    SetWindowTextW(hTextOutput, L"TEST");


   //wchar_t textname[100];
   // GetWindowTextW(hTextInput, textname, 100);
   // saveBMP((const char *)textname, imgWIDTH, imgHEIGHT, grid, maxIter);

    //saveBMP("test0.bmp", imgWIDTH, imgHEIGHT, grid, maxIter);


    // Initialise les chaînes globales
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_MANDELBROT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

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

void saveBMP();

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


void addControls(HWND hWnd) {
    //LABEL
    hTextOutput = CreateWindowW(L"static", L"Enter text here :", WS_VISIBLE | WS_CHILD, 10, 10, 200, 200, hWnd, NULL, NULL, NULL);
    //TEXT INPUT
    hTextInput = CreateWindowW(L"edit", L"Image1.bmp", WS_VISIBLE | WS_CHILD | WS_BORDER, 10, 210, 200, 200, hWnd, NULL, NULL, NULL);
    //SECTION GPU
    //hSection = CreateDIBSection(hdc, )
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
    case WM_CREATE:
        addControls(hWnd);
        break;
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
            hdc = BeginPaint(hWnd, &ps);
            // TODO: Ajoutez ici le code de dessin qui utilise hdc...

            /*
            int i, j, iter = 0;
            if (grid != NULL) {
                for (j = 0; j < imgHEIGHT; j++) {
                    for (i = 0; i < imgWIDTH; i++) {
                        iter++;
                        int mod = (unsigned char)*(grid + iter);
                        int r, g, b = 0;
                        switch (mod)
                        {
                                case 0: r = 66; g = 30; b = 15; break;
                                case 1: r = 25; g = 7; b = 26; break;
                                case 2: r = 9; g = 1; b = 47; break;
                                case 3: r = 4; g = 4; b = 73; break;
                                case 4: r = 0; g = 7; b = 100; break;
                                case 5: r = 12; g = 44; b = 138; break;
                                case 6: r = 24; g = 82; b = 177; break;
                                case 7: r = 57; g = 125; b = 209; break;
                                case 8: r = 134; g = 181; b = 229; break;
                                case 9: r = 211; g = 236; b = 248; break;
                                case 10: r = 241; g = 233; b = 191; break;
                                case 11: r = 248; g = 201; b = 95; break;
                                case 12: r = 254; g = 170; b = 0; break;
                                case 13: r = 204; g = 128; b = 0; break;
                                case 14: r = 153; g = 87; b = 0; break;
                                case 15: r = 106; g = 52; b = 3; break;
                        }
                        SetPixel(hdc, gridOffsetX + i, gridOffsetY + j, RGB(r, g, b));

                        
                    }
                }
            }
            */

            StretchDIBits(hdc, gridOffsetX, gridOffsetY, imgWIDTH, imgHEIGHT, 0, 0, imgWIDTH, imgHEIGHT, grid, &bitmap_info, DIB_RGB_COLORS, SRCCOPY);


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
