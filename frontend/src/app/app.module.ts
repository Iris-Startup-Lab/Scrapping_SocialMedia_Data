console.log('Comenzando el modulo'); // Para depuración
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http'; // Import HttpClientModule
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms'; 
//import { AppRoutingModule } from './app-routing.module'; // Se elimina porque fue cambiado por el routes
import { AppComponent } from './app.component';
import { ScrapingComponent } from './scraping/scraping.component';
import { AnalisisComponent } from './analisis/analisis.component';
import { ChatComponent } from './chat/chat.component';
import { SharedModule } from './shared/shared.module';
import { routes } from './app.routes'; // Importando el módulo de rutas


@NgModule({
  // Declaramos app component y demás partes de la aplicación
  declarations: [
    //AppComponent,
    //ScrapingComponent,
    //AnalisisComponent,
    //ChatComponent
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routes),
    //AppRoutingModule,
    //AppRoutes,
    HttpClientModule, // Add HttpClientModule to imports
    SharedModule, 
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }

console.log('Pasó el modulo'); // Para depuración