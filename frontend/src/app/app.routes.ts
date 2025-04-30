//import { Routes } from '@angular/router';

//export const routes: Routes = [];
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ScrapingComponent } from './scraping/scraping.component';
import { ChatComponent } from './chat/chat.component';

export const routes: Routes = [
  { path: 'scraping', component: ScrapingComponent },
  { path: 'chat', component: ChatComponent},
  { path: '', redirectTo: '/scraping', pathMatch: 'full' } // Redirige la raíz a /scraping

  // ... otras rutas ...
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }

console.log('Pasó el routes'); // Para depuración