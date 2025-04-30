import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter, Routes } from '@angular/router';
import { ScrapingComponent } from './scraping/scraping.component';
import { ChatComponent } from './chat/chat.component';
import { provideHttpClient } from '@angular/common/http';
//import { routes } from './app.routes';

//import { ScrapingComponent } from './scraping/scraping.component';

export const routes: Routes = [
  { path: 'scraping', component: ScrapingComponent },
  { path: 'chat', component: ChatComponent }, 
  { path: '', redirectTo: '/scraping', pathMatch: 'full' },
  // ... tus otras rutas
];

export const appConfig: ApplicationConfig = {
  providers: [provideRouter(routes), provideHttpClient()]
  //providers: [provideZoneChangeDetection({ eventCoalescing: true }), provideRouter(routes)]
};

console.log('Pasó el config'); // Para depuración
